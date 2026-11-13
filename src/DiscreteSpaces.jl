abstract type DiscreteSpace end

get_coords(f::DiscreteSpace) = @abstractmethod
num_coords(f::DiscreteSpace) = length(get_coords(f))

"""
    get_points(f) -> AbstractVector{<:Point}

The point set underlying `f`. It coincides with [`get_coords`](@ref) for every
space but [`InputSpace`](@ref), whose `get_coords` returns the `(D,N,K)` array
fed to the network (the points are then read off its first sample).
"""
get_points(f) = get_coords(f)

get_coords(x::AbstractVector{<:Point}) = x

function get_coords(f::SingleFieldFESpace)
  x = get_free_dof_coordinates(f)
  masks = get_node_masks(f)
  view(x,masks)
end

function get_coords(f::MultiFieldFESpace)
  lcoords = map(get_coords,f)
  ∪(lcoords...)
end

abstract type TrainingSpace <: DiscreteSpace end

fetch_data(d::DataFormatter,f::TrainingSpace) = @abstractmethod

struct InputSpace{A<:AbstractInputData,B} <: TrainingSpace
  coords::A
  features::B
end

function InputSpace(x::AbstractVector{<:Point},a::FeatureBundle)
  coords = points_to_coords(x,num_params(a))
  features = _get_features(a,x)
  InputSpace(coords,features)
end

function InputSpace(space,args...)
  x = get_coords(space)
  InputSpace(x,args...)
end

get_coords(f::InputSpace) = f.coords
get_features(f::InputSpace) = f.features
num_coords(f::InputSpace) = size(f.coords,2)

get_points(f::InputSpace) = coords_to_points(f.coords)

function fetch_data(d::DataFormatter,f::InputSpace)
  input = (coords=f.coords,features=f.features)
  fetch_data(d,input)
end

struct OutputSpace{A,B} <: TrainingSpace
  space::A
  values::B
end

function OutputSpace(uh::SingleFieldParamFEFunction)
  f = get_fe_space(uh)
  x = get_coords(f)
  values = _get_values(uh,x)
  OutputSpace(f,values)
end

function OutputSpace(space)
  OutputSpace(zero(space))
end

get_coords(f::OutputSpace) = get_coords(f.space)
get_values(f::OutputSpace) = f.values

function fetch_data(d::DataFormatter,f::OutputSpace)
  output = f.values
  fetch_data(d,output)
end

function to_element(f::OutputSpace,data)
  u = _free_values(f.space,data)
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

get_coords(f::FunctionSpace) = f.coords

function get_measure(f::FunctionSpace{<:FESpace};factor=2)
  trian = get_triangulation(f.space)
  degree = factor*get_polynomial_order(f.space)
  Measure(trian,degree)
end

"""
    struct GraphInducedSpace{A<:AbstractGraph,B<:AbstractVector{<:Point},C<:AbstractVector{Int},D<:AbstractVector{TData}} <: DiscreteSpace
      graph::A
      coords::B
      src_ids::C
      dst_ids::C
      edge_weights::D
    end

No precomputed/cached scatter structure needed (see `scatter_data` in
`GraphsInterface.jl`): it's built on `NNlib.scatter`, which doesn't need to stay off
an active Reactant/Enzyme trace's call stack the way the old `setindex!`-based dense
incidence matrix did, so there's nothing left to precompute here at construction time.
"""
struct GraphInducedSpace{A<:AbstractGraph,B<:AbstractVector{<:Point},C<:AbstractVector{Int},D<:AbstractVector{TData}} <: DiscreteSpace
  graph::A
  coords::B
  src_ids::C
  dst_ids::C
  edge_weights::D
end

function GraphInducedSpace(space;kwargs...)
  coords = get_coords(space)
  graph = build_graph(coords;kwargs...)
  es = edges(graph)
  src_ids,dst_ids = map(es) do e
    (src(e),dst(e))
  end |> tuple_of_arrays
  w = TData.(get_edge_weights(graph))
  wmax = maximum(w;init=zero(TData))
  wmax > zero(TData) && (w ./= wmax)
  GraphInducedSpace(graph,coords,src_ids,dst_ids,w)
end

get_graph(space::GraphInducedSpace) = space.graph
get_coords(space::GraphInducedSpace) = space.coords
get_src_ids(space::GraphInducedSpace) = space.src_ids
get_dst_ids(space::GraphInducedSpace) = space.dst_ids
get_edge_ids(space::GraphInducedSpace) = (get_src_ids(space),get_dst_ids(space))
get_edge_weights(space::GraphInducedSpace) = space.edge_weights

# multiple spaces

"""
    struct MultipleSpaces{N,A<:DiscreteSpace,B<:NTuple{N,DiscreteSpace},C<:Table,D<:Union{Nothing,AbstractVector{Int}}} <: DiscreteSpace
      gspace::A
      lspaces::B
      l_to_lcoord_to_coords::C
      l_to_sample_offset::D
    end

`N` local spaces `lspaces` glued into the global space `gspace`, which lives on the
union of the local point sets. Covers two scenarios:

- **one PDE on a multi-field FE space** (`l_to_sample_offset === nothing`): every
  local space (one per field) sees the same samples. Training data is laid out on the
  global points, with channels concatenated field by field (a point only some fields
  reach is zero-filled in every other field's channels);
- **several PDEs** (e.g. on a moving geometry, `l_to_sample_offset isa AbstractVector`):
  every local space comes with its own samples, and the global samples
  `l_to_sample_offset[l]:l_to_sample_offset[l+1]-1` are those of `lspaces[l]`
  (zero-filled outside of `lspaces[l]`'s points).

`l_to_lcoord_to_coords[l]` maps the points of `lspaces[l]` to those of `gspace`.

Built by [`InputSpace`](@ref)/[`OutputSpace`](@ref)/[`FunctionSpace`](@ref)/
[`GraphInducedSpace`](@ref) out of a `MultiFieldFESpace`/`MultiFieldParamFEFunction`
(first scenario), or out of several input/output spaces/`ParamFEFunction`s (second
scenario). `FunctionSpace(f::MultipleSpaces)`/`GraphInducedSpace(f::MultipleSpaces)`
build the integration spaces matching the layout (and samples) of `f`.
"""
struct MultipleSpaces{N,A<:DiscreteSpace,B<:NTuple{N,DiscreteSpace},C<:Table,D<:Union{Nothing,AbstractVector{Int}}} <: DiscreteSpace
  gspace::A
  lspaces::B
  l_to_lcoord_to_coords::C
  l_to_sample_offset::D
end

function MultipleSpaces(gspace::DiscreteSpace,lspaces::Tuple{Vararg{DiscreteSpace}})
  l_to_lcoord_to_coords = _lcoord_to_coords_map(map(get_points,lspaces),get_points(gspace))
  MultipleSpaces(gspace,lspaces,l_to_lcoord_to_coords,nothing)
end

get_coords(space::MultipleSpaces) = get_coords(space.gspace)
get_points(space::MultipleSpaces) = get_points(space.gspace)
num_coords(space::MultipleSpaces) = num_coords(space.gspace)
get_graph(space::MultipleSpaces) = get_graph(space.gspace)

Base.length(space::MultipleSpaces) = length(space.lspaces)
Base.getindex(space::MultipleSpaces,i::Int) = space.lspaces[i]
Base.iterate(space::MultipleSpaces,state...) = iterate(space.lspaces,state...)

const MultipleInputSpaces = MultipleSpaces{N,<:InputSpace} where N
# TODO this concept is a bit weird: do we ever evaluate on multiple PDEs?
# I guess yes, in principle, but we may want to eliminate this in the future
const MultipleOutputSpaces = MultipleSpaces{N,<:OutputSpace} where N

const AbstractInputSpace = Union{InputSpace,MultipleInputSpaces}
const AbstractOutputSpace = Union{OutputSpace,MultipleOutputSpaces}

fetch_data(d::DataFormatter,f::MultipleSpaces) = fetch_data(d,f.gspace)

function to_element(f::MultipleOutputSpaces,data::AbstractInputData)
  _to_element(f,f.l_to_sample_offset,data)
end

# multi-field constructors: shared samples, channels stacked field by field

function InputSpace(space::MultiFieldFESpace,a::FeatureBundle)
  x = get_coords(space)
  fields = Tuple(space.spaces)
  gspace = InputSpace(x,a)
  lspaces = map(s -> InputSpace(s,a),fields)
  l_to_lcoord_to_coords = _lcoord_to_coords_map(map(get_coords,fields),x)
  MultipleSpaces(gspace,lspaces,l_to_lcoord_to_coords,nothing)
end

function OutputSpace(uh::MultiFieldParamFEFunction)
  space = get_fe_space(uh)
  x = get_coords(space)
  lspaces = Tuple(map(OutputSpace,uh))
  l_to_lcoord_to_coords = _lcoord_to_coords_map(map(get_coords,lspaces),x)
  values = cat_channels(map(get_values,lspaces),l_to_lcoord_to_coords,length(x))
  MultipleSpaces(OutputSpace(space,values),lspaces,l_to_lcoord_to_coords,nothing)
end

function FunctionSpace(space::MultiFieldFESpace)
  gspace = FunctionSpace(space,get_coords(space))
  lspaces = map(FunctionSpace,Tuple(space.spaces))
  MultipleSpaces(gspace,lspaces)
end

function GraphInducedSpace(space::MultiFieldFESpace;kwargs...)
  gspace = GraphInducedSpace(get_coords(space);kwargs...)
  lspaces = map(s -> GraphInducedSpace(s;kwargs...),Tuple(space.spaces))
  MultipleSpaces(gspace,lspaces)
end

# multi-PDE constructors: one block of samples per local space

"""
    InputSpace(lspaces::InputSpace...) -> MultipleSpaces
    OutputSpace(lspaces::OutputSpace...) -> MultipleSpaces
    OutputSpace(uhs::ParamFEFunction...) -> MultipleSpaces

Stacks the samples of several PDEs (one per local space) on the union of their
points; see [`MultipleSpaces`](@ref). Input and output spaces built from local spaces
listed in the same order share the same global layout.
"""
function InputSpace(lspaces::InputSpace...)
  lpoints = map(get_points,lspaces)
  x = collect(∪(lpoints...))
  l_to_lcoord_to_coords = _lcoord_to_coords_map(lpoints,x)
  features,l_to_sample_offset = cat_samples(map(get_features,lspaces),l_to_lcoord_to_coords,length(x))
  coords = points_to_coords(x,size(features,3))
  gspace = InputSpace(coords,features)
  MultipleSpaces(gspace,lspaces,l_to_lcoord_to_coords,l_to_sample_offset)
end

function OutputSpace(lspaces::OutputSpace...)
  lpoints = map(get_points,lspaces)
  x = collect(∪(lpoints...))
  l_to_lcoord_to_coords = _lcoord_to_coords_map(lpoints,x)
  values,l_to_sample_offset = cat_samples(map(get_values,lspaces),l_to_lcoord_to_coords,length(x))
  gspace = OutputSpace(x,values)
  MultipleSpaces(gspace,lspaces,l_to_lcoord_to_coords,l_to_sample_offset)
end

OutputSpace(uhs::ParamFEFunction...) = OutputSpace(map(OutputSpace,uhs)...)

# integration spaces matching the layout (and the samples) of a multiple space

function FunctionSpace(f::MultipleSpaces)
  space = _fe_space(f.gspace)
  gspace = FunctionSpace(space,get_coords(space))
  lspaces = map(s -> FunctionSpace(_fe_space(s)),f.lspaces)
  MultipleSpaces(gspace,lspaces,f.l_to_lcoord_to_coords,f.l_to_sample_offset)
end

function GraphInducedSpace(f::MultipleSpaces;kwargs...)
  gspace = GraphInducedSpace(get_points(f.gspace);kwargs...)
  lspaces = map(s -> GraphInducedSpace(get_points(s);kwargs...),f.lspaces)
  MultipleSpaces(gspace,lspaces,f.l_to_lcoord_to_coords,f.l_to_sample_offset)
end

# the space a FunctionSpace is built on
_fe_space(f::DiscreteSpace) = get_points(f)
_fe_space(f::OutputSpace) = f.space
_fe_space(f::FunctionSpace) = f.space

# norms

"""
    assemble_norm(op::AssembleOperator,f) -> AbstractMatrix

Gram matrix of the `GridapROMs` norm `op` (`ℓ2()`, `L2()`, `H1()`, ...) on the
point layout of the `(D,N,K)` data living on `f`, rather than on `f`'s dofs:
- single field: `(N,N)`, restricted to one representative dof per point (the
  norms are component-wise, so every component shares this same matrix);
- multi-field: `(D*N,D*N)`, block-diagonal with one block per channel, each
  field's matrix being extended by zero from its own points to the union of
  every field's points (the layout of a multi-field [`MultipleSpaces`](@ref)).
"""
assemble_norm(op::AssembleOperator,f::FunctionSpace) = assemble_norm(op,f.space)
assemble_norm(op::AssembleOperator,f::MultipleSpaces) = assemble_norm(op,f.gspace)

function assemble_norm(op::AssembleOperator,f::SingleFieldFESpace)
  mat = _assemble_dof_norm(op,f)
  msk = get_node_masks(f)
  return TData.(mat[msk,msk])
end

function assemble_norm(op::AssembleOperator,f::MultiFieldFESpace)
  assemble_norm(BlockOperator(ntuple(_ -> op,length(f.spaces))),f)
end

function assemble_norm(op::BlockOperator,f::MultiFieldFESpace)
  @check length(op.op) == length(f.spaces) "Wrong number of norms for the MultiFieldFESpace"
  x = get_coords(f)
  n = length(x)
  blocks = []
  for (opi,spacei) in zip(op.op,f.spaces)
    Xi = assemble_norm(opi,spacei)
    pos = indexin(get_coords(spacei),x)
    I,J,V = findnz(Xi)
    Xi_global = sparse(pos[I],pos[J],V,n,n)
    for _ in 1:maximum(get_free_dof_to_comp(spacei))
      push!(blocks,Xi_global)
    end
  end
  blockdiag(blocks...)
end

_assemble_dof_norm(op::AssembleOperator,f::SingleFieldFESpace) = RBSteady._assemble_operator(op,f,f)
_assemble_dof_norm(::ℓ2,f::SingleFieldFESpace) = sparse(I(num_free_dofs(f)))

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

for F in (:get_free_dof_to_comp,:get_free_dof_to_point)
  @eval begin
    function $F(f)
      @notimplementedif !hasfield(typeof(f),:space)
      $F(getfield(f,:space))
    end
  end
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

function _free_values(f::FESpace,data::AbstractInputData)
  msg = "Data does not match the space's dof count"
  comps = get_free_dof_to_comp(f)
  points = get_free_dof_to_point(f)
  ids = findall(get_node_masks(f)) # representative dof ids
  point_to_pos = Dict(points[dof] => pos for (pos,dof) in enumerate(ids))
  @check size(data,1)*size(data,2) == length(comps) msg
  T = eltype2(get_vector_type(f))
  u = zeros(T,length(comps),size(data,3))
  @inbounds @views for dof in eachindex(comps)
    pos = point_to_pos[points[dof]]
    u[dof,:] .= data[comps[dof],pos,:]
  end
  u
end

function _lcoord_to_coords_map(lcoords,coords)
  coord_to_id = Dict(x => i for (i,x) in enumerate(coords))
  ptrs = ones(Int32,length(lcoords)+1)
  for (l,lc) in enumerate(lcoords)
    ptrs[l+1] = ptrs[l] + length(lc)
  end
  data = zeros(Int,ptrs[end]-1)
  for (l,lc) in enumerate(lcoords)
    for (i,x) in enumerate(lc)
      data[ptrs[l]+i-1] = coord_to_id[x]
    end
  end
  Table(data,ptrs)
end

# one FE function on the multi-field space: channels are stacked field by field
function _to_element(f::MultipleOutputSpaces,::Nothing,data::AbstractInputData)
  msg = "Data does not match the space's dof count"
  T = eltype2(get_vector_type(f.gspace.space))
  u = zeros(T,num_free_dofs(f.gspace.space),size(data,3))
  lvals = map(get_values,f.lspaces)
  offsets = compute_offsets(lvals;axis=1)
  dof_offsets = compute_field_offsets(f.gspace.space)
  cache = array_cache(f.l_to_lcoord_to_coords)
  @inbounds @views for (l,lspace) in enumerate(f.lspaces)
    lcoord_to_coords = getindex!(cache,f.l_to_lcoord_to_coords,l)
    ldata = get_channel(data,offsets[l]:offsets[l+1]-1,lcoord_to_coords)
    comps = get_free_dof_to_comp(lspace)
    points = get_free_dof_to_point(lspace)
    ids = findall(get_node_masks(lspace)) # representative dof ids
    point_to_pos = Dict(points[dof] => pos for (pos,dof) in enumerate(ids))
    @check size(ldata,1)*size(ldata,2) == length(comps) msg
    for dof in eachindex(comps)
      pos = point_to_pos[points[dof]]
      u[dof_offsets[l]+dof,:] .= ldata[comps[dof],pos,:]
    end
  end
  FEFunction(f.gspace.space,ConsecutiveParamArray(u))
end

# one FE function per PDE, each evaluated on its own samples
function _to_element(f::MultipleOutputSpaces,l_to_sample_offset,data::AbstractInputData)
  @check size(data,3) == last(l_to_sample_offset)-1 "Data does not match the number of samples of the output space"
  ntuple(length(f)) do l
    samples = _get_range(l_to_sample_offset,l)
    @views ldata = data[:,f.l_to_lcoord_to_coords[l],samples]
    to_element(f.lspaces[l],ldata)
  end
end