# case 1: only 1 space
# case 2: multiple spaces, but all have the same mesh
# -> several cell fields, only one measure
# case 3: multiple spaces, each with a different mesh
# -> several cell fields, several measures

abstract type DiscreteSpace end

get_coords(f::DiscreteSpace) = @abstractmethod
num_coords(f::DiscreteSpace) = length(get_coords(f))

function get_coordinates(f::SingleFieldFESpace)
  x = get_free_dof_coordinates(f)
  masks = get_node_masks(f)
  view(x,masks)
end

"""
    get_coordinates(f::MultiFieldFESpace)

When every field shares the exact same point set (same mesh, same order), this
is just `get_coordinates(first(f))`. Otherwise (e.g. a Taylor-Hood pair, where
the quadratic velocity space has more points than the linear pressure one), it's
the *union* of every field's own points: a `ParamFEFunction` can be evaluated at
any point in its domain, not only its own dofs (ordinary FE interpolation, the
same machinery `_get_values` already uses for a field's own masked dof
coordinates), so `_get_values(::MultiFieldParamFEFunction,...)` (`DataInterface.jl`)
can evaluate every field at this shared, dense point set directly and
concatenate channel-wise -- no padding needed, since every field is now
genuinely defined (albeit, at a non-native point, by interpolation rather than
an exact dof) at every point.
"""
function get_coordinates(f::MultiFieldFESpace)
  lcoords = map(get_coordinates,f)
  x = first(lcoords)
  all(==(x),lcoords) && return x
  collect(∪(map(Set,lcoords)...))
end

abstract type TrainingSpace <: DiscreteSpace end

fetch_data(d::DataFormatter,f::TrainingSpace) = @abstractmethod

struct InputSpace{A<:InputData,B<:InputData} <: TrainingSpace
  coords::B
  features::A
end

function InputSpace(x::AbstractVector{<:Point},a::FeatureBundle)
  coords = _get_coords(x,num_params(a))
  features = _get_features(a,x)
  InputSpace(coords,features)
end

function InputSpace(space,a::FeatureBundle)
  x = get_coords(space)
  InputSpace(x,a)
end

get_coords(f::InputSpace) = f.coords
get_features(f::InputSpace) = f.features

function fetch_data(d::DataFormatter,f::InputSpace)
  input = (coords=f.coords,features=f.features)
  fetch_data(d,input)
end

struct OutputSpace{A,B<:InputData} <: TrainingSpace
  space::A
  values::B
end

function OutputSpace(uh::ParamFEFunction)
  f = get_fe_space(uh)
  x = get_coordinates(f)
  values = _get_values(uh,x)
  OutputSpace(f,values)
end

function OutputSpace(space)
  OutputSpace(zero(space))
end

function fetch_data(d::DataFormatter,f::OutputSpace)
  output = f.values
  fetch_data(d,output)
end

function to_element(f::OutputSpace,data::InputData)
  space = f.space
  u = _free_values(space,data,f.values.ranges)
  pu = ConsecutiveParamArray(u)
  FEFunction(space,pu)
end

# `ranges` (from the OutputSpace's own RangedInputData, not from `data`: `data` here
# is typically a plain prediction fresh out of the model, not itself sized) is only
# meaningful for a MultiFieldFESpace, so a single field just ignores it.
_free_values(space,data::InputData,ranges) = _free_values(space,data)

function _free_values(space,data::InputData)
  comps = get_free_dof_to_comp(space)
  points = get_free_dof_to_point(space)
  # `data`'s 2nd axis is indexed in "one representative dof per point" order (see
  # get_coordinates' masking), not by raw point id, so we need the point ->
  # position-in-that-order map
  rep_dofs = findall(get_node_masks(space))
  point_to_pos = Dict(points[dof] => pos for (pos,dof) in enumerate(rep_dofs))
  @check size(data,1)*size(data,2) == length(comps) "Data does not match the output space's dof count"
  T = eltype2(get_dirichlet_dof_values(space))
  u = zeros(T,length(comps),size(data,3))
  @inbounds @views for dof in eachindex(comps)
    pos = point_to_pos[points[dof]]
    u[dof,:] .= data[comps[dof],pos,:]
  end
  u
end

"""
    _free_values(space::MultiFieldFESpace,data::InputData,ranges::NTuple{N,AbstractVector})

`data`'s point axis is indexed by `get_coordinates(space)` -- when fields don't
share points, that's their *union* (see `get_coordinates(::MultiFieldFESpace)`),
not any single field's own points, so field `i`'s own masked points are only a
subset of it, in some order (`indexin`, matching by coordinate value, the same
way `_coords_info` does for `MultiGraphInducedSpace`): the generic single-field
reconstruction expects its data's point axis to correspond 1:1 (in its own
order) to `get_coordinates(space.spaces[i])`, so that subset needs to be pulled
out first.
"""
function _free_values(space::MultiFieldFESpace,data::InputData,ranges::NTuple{N,AbstractVector}) where N
  combined = get_coordinates(space)
  us = ntuple(N) do i
    idx = indexin(get_coordinates(space.spaces[i]),combined)
    _free_values(space.spaces[i],view(data,ranges[i],idx,:))
  end
  reduce(vcat,us)
end

struct FunctionSpace{A,B} <: DiscreteSpace
  space::A
  coords::B
end

function FunctionSpace(space)
  coords = get_coordinates(space)
  FunctionSpace(space,coords)
end

function FunctionSpace(space::MultiFieldFESpace)
  trian = get_triangulation(first(space))
  if !all(s -> trian === get_triangulation(s),space)
    lcoords = map(get_coordinates,space)
    sets = map(Set,lcoords)
    !isempty(∩(sets...)) && return FunctionSpace(space...)
  end
  coords = get_coordinates(space)
  FunctionSpace(space,coords)
end

function FunctionSpace(spaces...)
  @warn "Returning the space induced by the graph of the union of the underlying meshes"
  MultiGraphInducedSpace(spaces...)
end

get_coords(f::FunctionSpace) = f.coords

function get_measure(f::FunctionSpace{<:FESpace};factor=2)
  trian = get_triangulation(f.space)
  degree = factor*get_polynomial_order(f.space)
  Measure(trian,degree)
end

abstract type AbstractGraphInducedSpace <: DiscreteSpace end

get_graph(space::AbstractGraphInducedSpace) = @abstractmethod

for f in (:gather_data,:scatter_data,:attention)
  @eval $f(vals::AbstractArray,space::AbstractGraphInducedSpace) = $f(vals,get_graph(space))
end

"""
    struct GraphInducedSpace{A<:Graphs.AbstractGraph,B<:AbstractVector{<:Point},C<:AbstractMatrix} <: AbstractGraphInducedSpace
      graph::A
      coords::B
      scatter_matrix::C
    end

`scatter_matrix` is `scatter_data`'s dense incidence matrix (see `GraphsInterface.jl`),
precomputed once here -- plain, untraced Julia, at space-construction time -- rather
than rebuilt on every `scatter_data` call. That matters specifically because
`scatter_data` normally gets called from deep inside an active Reactant/Enzyme
training trace (via `_integration(::GraphMeasure)`); building the matrix there, even
though its inputs don't depend on any traced value, still hits Reactant's "scalar
indexing disallowed" (its overlay intercepts the whole nested call stack, not just
operations that touch traced values).
"""
struct GraphInducedSpace{A<:AbstractGraph,B<:AbstractVector{<:Point},C<:AbstractMatrix} <: AbstractGraphInducedSpace
  graph::A
  coords::B
  scatter_matrix::C
  function GraphInducedSpace(graph::A,coords::B) where {A<:Graphs.AbstractGraph,B<:AbstractVector{<:Point}}
    @check nv(graph) == length(coords) "GraphInducedSpace needs exactly one coordinate per vertex"
    S = _scatter_matrix(TData,graph)
    new{A,B,typeof(S)}(graph,coords,S)
  end
end

function GraphInducedSpace(space;kwargs...)
  coords = get_coords(space)
  build_graph(coords;kwargs...)
end

get_graph(space::GraphInducedSpace) = space.graph
get_coords(space::GraphInducedSpace) = space.coords
scatter_data(vals::InputData,space::GraphInducedSpace) = _scatter_apply(vals,space.scatter_matrix)

struct MultiGraphInducedSpace{N,A<:AbstractGraph,B<:AbstractVector{<:Point},C<:Table} <: AbstractGraphInducedSpace
  graph::A
  graphs::NTuple{N,A}
  coords::B
  node_to_lnode::C
end

function MultiGraphInducedSpace(spaces...;kwargs...)
  lcoords = map(get_coords,spaces)
  lgraphs = map(x -> build_graph(x;kwargs...),lcoords)
  coords,node_to_lnode = _coords_info(lcoords)
  graph = build_graph(coords;kwargs...)
  MultiGraphInducedSpace(graph,lgraphs,coords,node_to_lnode)
end

get_graph(space::MultiGraphInducedSpace) = space.graph
get_coords(space::MultiGraphInducedSpace) = space.coords

# utils

"""
    get_free_dof_to_comp(f)
    get_free_dof_to_point(f)

Metadata-independent replacements for reading off a `SingleFieldFESpace`'s
`NodeToDofGlue` (only ever built by Gridap for order-1 Lagrangian spaces; any
higher order, e.g. the quadratic velocity space in a Taylor-Hood Stokes pair,
gets the more general `CompressedCellConformity` instead, which has no
precomputed dof/node glue arrays). Both are derived generically via a cell
traversal, so they work for any `UnconstrainedFESpace` regardless of which
`CellConformity` it was built with.

`get_free_dof_to_point` plays the role `NodeToDofGlue.free_dof_to_node` used to:
rather than trying to reconstruct Gridap's internal mesh-node numbering (which
isn't well-defined once higher-order dofs, e.g. edge midpoints, are involved),
free dofs are grouped directly by their (exact) physical coordinate -- dofs at
the same point (e.g. one per vector component) necessarily share one.
"""
get_node_masks(f) = _first_dof_per_point(get_free_dof_to_point(f))

function get_free_dof_to_comp(f)
  @notimplementedif !hasfield(typeof(f),:space)
  get_free_dof_to_comp(getfield(f,:space))
end

function get_free_dof_to_comp(f::UnconstrainedFESpace)
  cell_dof_ids = get_cell_dof_ids(f)
  cell_dof_to_comp = lazy_map(get_dof_to_comp,get_data(get_fe_dof_basis(f)))
  comps = zeros(Int16,num_free_dofs(f))
  cache_ids = array_cache(cell_dof_ids)
  cache_comp = array_cache(cell_dof_to_comp)
  for cell in eachindex(cell_dof_ids)
    ids = getindex!(cache_ids,cell_dof_ids,cell)
    cps = getindex!(cache_comp,cell_dof_to_comp,cell)
    for (ldof,gdof) in enumerate(ids)
      gdof > 0 && (comps[gdof] = cps[ldof])
    end
  end
  comps
end

function get_free_dof_to_point(f)
  @notimplementedif !hasfield(typeof(f),:space)
  get_free_dof_to_point(getfield(f,:space))
end

function get_free_dof_to_point(f::UnconstrainedFESpace)
  x = get_free_dof_coordinates(f)
  dict = Dict{eltype(x),Int32}()
  point = zeros(Int32,length(x))
  count = Int32(0)
  @inbounds for dof in eachindex(x)
    xi = x[dof]
    if !haskey(dict,xi)
      count += Int32(1)
      dict[xi] = count
    end
    point[dof] = dict[xi]
  end
  point
end

function _first_dof_per_point(points::AbstractVector{<:Integer})
  touched = falses(maximum(points,init=0))
  mask = falses(length(points))
  @inbounds for (dof,p) in enumerate(points)
    if !touched[p]
      touched[p] = true
      mask[dof] = true
    end
  end
  mask
end

function _a_to_b(a,b)
  a2b = zeros(Int,length(b))
  for (i,bi) in enumerate(b)
    a2b[i] = findfirst(x -> x == bi,a)
  end
  a2b
end

function _coord_to_lcoords_map(coords,lcoords)
  data = zeros(Int,sum(map(length,lcoords)))
  ptrs = ones(TData,length(lcoords)+1)
  for (i,lc) in enumerate(lcoords)
    ptrs[i+1] = ptrs[i] + length(lc)
    @views data[ptrs[i]:(ptrs[i+1]-1)] .= _a_to_b(coords,lc)
  end
  Table(data,ptrs)
end

function _coords_info(lcoords)
  sets = map(Set,lcoords)
  coords = ∪(sets...)
  coord_to_lcoords = _coord_to_lcoords_map(coords,lcoords)
  return coords,coord_to_lcoords
end