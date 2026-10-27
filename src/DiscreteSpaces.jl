# case 1: only 1 space
# case 2: multiple spaces, but all have the same mesh
# -> several cell fields, only one measure
# case 3: multiple spaces, each with a different mesh
# -> several cell fields, several measures

abstract type DiscreteSpace end

get_coords(f::DiscreteSpace) = @abstractmethod
num_coords(f::DiscreteSpace) = length(get_coords(f))

function get_coords(f::SingleFieldFESpace)
  x = get_free_dof_coordinates(f)
  masks = get_node_masks(f)
  view(x,masks)
end

function get_coords(f::MultiFieldFESpace)
  map(get_coords,f)
end

abstract type TrainingSpace <: DiscreteSpace end

fetch_data(d::DataFormatter,f::TrainingSpace) = @abstractmethod

struct InputSpace{A<:AbstractInputData,B<:AbstractInputData} <: TrainingSpace
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
  x = get_coords(f)
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

function to_element(f::OutputSpace,data::AbstractInputData)
  u = _free_values(f,data)
  pu = ConsecutiveParamArray(u)
  FEFunction(f.space,pu)
end

struct FunctionSpace{A,B} <: DiscreteSpace
  space::A
  coords::B
end

function FunctionSpace(space)
  coords = get_coords(space)
  FunctionSpace(space,coords)
end

function FunctionSpace(space::MultiFieldFESpace)
  trian = get_triangulation(first(space))
  if !all(s -> trian === get_triangulation(s),space)
    lcoords = map(get_coords,space)
    sets = map(Set,lcoords)
    !isempty(∩(sets...)) && return FunctionSpace(space...)
  end
  coords = get_coords(space)
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
    struct GraphInducedSpace{A<:AbstractGraph,B<:AbstractVector{<:Point}} <: AbstractGraphInducedSpace
      graph::A
      coords::B
    end

No precomputed/cached scatter structure needed (see `scatter_data` in
`GraphsInterface.jl`): it's built on `NNlib.scatter`, which doesn't need to stay off
an active Reactant/Enzyme trace's call stack the way the old `setindex!`-based dense
incidence matrix did, so there's nothing left to precompute here at construction time.
"""
struct GraphInducedSpace{A<:AbstractGraph,B<:AbstractVector{<:Point}} <: AbstractGraphInducedSpace
  graph::A
  coords::B
end

function GraphInducedSpace(space;kwargs...)
  coords = get_coords(space)
  build_graph(coords;kwargs...)
end

get_graph(space::GraphInducedSpace) = space.graph
get_coords(space::GraphInducedSpace) = space.coords

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

function Base.iterate(space::MultiGraphInducedSpace)
  space,space_it = iterate(get_space(meas))
  weights,weights_it = iterate(get_weights(meas))
  if isnothing(space_it) || isnothing(weights_it)
    return nothing
  end
  (InducedMeasure(space,weights),(space_it,weights_it))
end

function Base.iterate(space::MultiGraphInducedSpace,state)
  space_it,weights_it = state
  space,space_it = iterate(get_space(meas),space_it)
  weights,weights_it = iterate(get_weights(meas),weights_it)
  if isnothing(space_it) || isnothing(weights_it)
    return nothing
  end
  (InducedMeasure(space,weights),(space_it,weights_it))
end

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
necessarily looks up every dof's coordinate against `get_coords(f)` -- the
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
  x = get_coords(f)
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
  ptrs = ones(Int32,length(lcoords)+1)
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

function _free_values(f::OutputSpace,data::AbstractInputData)
  msg = "Data does not match the output space's dof count"
  comps = get_free_dof_to_comp(f.space)
  points = get_free_dof_to_point(f.space)
  ids = findall(get_node_masks(f.space)) # representative dof ids
  point_to_pos = Dict(points[dof] => pos for (pos,dof) in enumerate(ids))
  @check size(data,1)*size(data,2) == length(comps) msg
  T = eltype2(get_vector_type(f.space))
  u = zeros(T,length(comps),size(data,3))
  @inbounds @views for dof in eachindex(comps)
    pos = point_to_pos[points[dof]]
    u[dof,:] .= data[comps[dof],pos,:]
  end
  u
end

function _free_values(f::OutputSpace{<:MultiFieldFESpace},data::AbstractInputData)
  caxis,paxis = _block_axes(f)
  us = ntuple(length(f.space)) do i
    fi = OutputSpace(f.space[i],f.values)
    vi = view(data,caxis[i]:caxis[i+1]-1,paxis[i]:paxis[i+1]-1,:)
    _free_values(fi,vi)
  end
  reduce(vcat,us)
end

function _block_axes(f::OutputSpace{<:MultiFieldFESpace})
  caxis,paxis, = f.values.axes
  return caxis,paxis
end