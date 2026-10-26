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

The *union* of every field's own points (each field's own masked dof
coordinates, see `get_coordinates(::SingleFieldFESpace)`) -- always, even when
every field happens to share the exact same point set (same mesh, same order),
in which case the union is simply that shared set. This is what lets a field be
evaluated at a point it doesn't natively own (e.g. a Taylor-Hood pair, where the
quadratic velocity space has more points than the linear pressure one): a
`ParamFEFunction` can be evaluated at any point in its domain via ordinary FE
interpolation (the same machinery `_get_values` already uses for a field's own
masked dof coordinates), so `_get_values(::MultiFieldParamFEFunction,...)`
(`DataInterface.jl`) can evaluate every field at this shared, dense point set
directly and concatenate channel-wise -- no padding needed, since every field is
now genuinely defined (albeit, at a non-native point, by interpolation rather
than an exact dof) at every point.
"""
function get_coordinates(f::MultiFieldFESpace)
  lcoords = map(get_coordinates,f)
  first(_union_with_positions(lcoords))
end

"""
    _union_with_positions(lcoords) -> (x,point_ranges)

Builds the union `x` of the given per-field coordinate lists in a single pass,
recording -- as a free byproduct of the very same pass that decides whether
each point is new or already present -- the position each field's own points
land at in `x`: `point_ranges[i][k]` is the position in `x` of `lcoords[i][k]`.
Deliberately not `collect(∪(map(Set,lcoords)...))` + a separate `indexin` pass
to recover `point_ranges` afterwards: that would redo, as a search against the
already-built `x`, exactly the membership check this construction needs anyway.
"""
function _union_with_positions(lcoords)
  T = eltype(first(lcoords))
  dict = Dict{T,Int}()
  x = T[]
  point_ranges = map(lcoords) do lc
    ids = zeros(Int,length(lc))
    for (k,xi) in enumerate(lc)
      ids[k] = get!(dict,xi) do
        push!(x,xi)
        length(x)
      end
    end
    ids
  end
  return x,point_ranges
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

struct OutputSpace{A,B<:RangedInputData} <: TrainingSpace
  space::A
  values::B
end

function OutputSpace(uh::ParamFEFunction)
  f = get_fe_space(uh)
  x = get_coordinates(f)
  values = _get_values(uh,x)
  OutputSpace(f,values)
end

"""
    OutputSpace(uh::MultiFieldParamFEFunction)

Specialised over the generic `OutputSpace(uh::ParamFEFunction)` to avoid an
unnecessary `indexin` search: `x` and each field's `point_ranges` are built
together here, in the one pass that already knows them (`_union_with_positions`),
and handed straight to `_get_values` -- instead of computing `x` alone and
leaving `_get_values` to recover `point_ranges` from it afterwards.
"""
function OutputSpace(uh::MultiFieldParamFEFunction)
  f = get_fe_space(uh)
  lcoords = map(get_coordinates,f)
  x,point_ranges = _union_with_positions(lcoords)
  values = _get_values(uh,x,point_ranges)
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
  u = _free_values(f,data)
  pu = ConsecutiveParamArray(u)
  FEFunction(f.space,pu)
end

function _free_values(f::OutputSpace,data::InputData)
  msg = "Data does not match the output space's dof count"
  comps = get_free_dof_to_comp(f.space)
  points = get_free_dof_to_point(f.space)
  ids = findall(get_node_masks(f.space)) # representative dof ids
  point_to_pos = Dict(points[dof] => pos for (pos,dof) in enumerate(ids))
  @check size(data,1)*size(data,2) == length(comps) msg
  T = eltype2(get_dirichlet_dof_values(f.space))
  u = zeros(T,length(comps),size(data,3))
  @inbounds @views for dof in eachindex(comps)
    pos = point_to_pos[points[dof]]
    u[dof,:] .= data[comps[dof],pos,:]
  end
  u
end

function _free_values(f::OutputSpace{<:MultiFieldFESpace},data::InputData)
  ranges = f.values.ranges
  point_ranges = f.values.point_ranges
  us = ntuple(length(ranges)) do i
    fi = OutputSpace(f.space[i],f.values)
    _free_values(fi,view(data,ranges[i],point_ranges[i],:))
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
function get_node_masks(f)
  points = get_free_dof_to_point(f)
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

"""
    get_free_dof_to_comp(f::MultiFieldFESpace)
    get_free_dof_to_point(f::MultiFieldFESpace)

Field-by-field concatenation of each field's own [`get_free_dof_to_comp`](@ref)/
[`get_free_dof_to_point`](@ref), using Gridap's own per-field dof offsets
(`compute_field_offsets`, valid for both `ConsecutiveMultiFieldStyle` and
`BlockMultiFieldStyle`) to place each field's dofs at their actual position in
the whole space's free-dof numbering. `get_free_dof_to_comp` keeps each dof's
component index local to its own field (not merged across fields -- a field's
component numbering is meaningless outside of it). `get_free_dof_to_point`
necessarily looks up every dof's coordinate against `get_coordinates(f)` -- the
*union* of every field's own points, since two different fields generally don't
share the same point set -- rather than against any single field's own points.
"""
function get_free_dof_to_comp(f::MultiFieldFESpace)
  comps = zeros(Int16,num_free_dofs(f))
  offsets = compute_field_offsets(f)
  for (i,space_i) in enumerate(f.spaces)
    comps_i = get_free_dof_to_comp(space_i)
    comps[offsets[i]+1:offsets[i]+length(comps_i)] .= comps_i
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
  @inbounds for (i,xi) in enumerate(x)
    if !haskey(dict,xi)
      count += Int32(1)
      dict[xi] = count
    end
    point[i] = dict[xi]
  end
  point
end

function get_free_dof_to_point(f::MultiFieldFESpace)
  x = get_coordinates(f)
  dict = Dict(xi => Int32(i) for (i,xi) in enumerate(x))
  point = zeros(Int32,num_free_dofs(f))
  offsets = compute_field_offsets(f)
  for (i,space_i) in enumerate(f.spaces)
    x_i = get_free_dof_coordinates(space_i)
    off = offsets[i]
    @inbounds for dof in eachindex(x_i)
      point[off+dof] = dict[x_i[dof]]
    end
  end
  point
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