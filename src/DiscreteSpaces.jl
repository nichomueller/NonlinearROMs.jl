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
  lcoords = map(get_coords,f)
  ∪(lcoords...)
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

function InputSpace(x::AbstractVector{<:Point},a::FeatureBundle...)
  InputSpace(x,merge_bundles(a...))
end

function InputSpace(space,args...)
  x = get_coords(space)
  InputSpace(x,a)
end

get_coords(f::InputSpace) = f.coords
get_features(f::InputSpace) = f.features

function fetch_data(d::DataFormatter,f::InputSpace)
  input = (coords=f.coords,features=f.features)
  fetch_data(d,input)
end

struct OutputSpace{A,B<:AbstractInputData} <: TrainingSpace
  space::A
  values::B
end

function OutputSpace(uh::SingleFieldParamFEFunction)
  f = get_fe_space(uh)
  x = get_coords(f)
  values = _get_values(uh,x)
  OutputSpace(f,values)
end

function OutputSpace(uh::MultiFieldParamFEFunction)
  spaces = map(OutputSpace,uh)
  join_spaces(spaces...)
end

function OutputSpace(uhs::ParamFEFunction...)
  spaces = map(OutputSpace,uhs)
  merge_spaces(spaces...)
end

function OutputSpace(space)
  OutputSpace(zero(space))
end

"""
    join_spaces(spaces::OutputSpace...) -> OutputSpace{<:MultiFieldFESpace}

Combines several single-field `OutputSpace`s -- sharing a triangulation, but not
necessarily their own point set (e.g. a Taylor-Hood velocity/pressure pair: the
lower-order field's points are a subset of the higher-order one's) -- into one
`OutputSpace` over their `MultiFieldFESpace`, with `values` laid out on the shared
*union* of every field's points ([`get_coords`](@ref), consistent with how
[`FunctionSpace`](@ref)/[`GraphInducedSpace`](@ref) build their measure/graph for
the same multi-field space), channels concatenated field-by-field. A point only
one field reaches is zero-filled in every other field's channels.
"""
function join_spaces(spaces::OutputSpace...)
  space = MultiFieldFESpace([s.space for s in spaces])
  x = get_coords(space)
  K = num_samples(first(spaces).values)
  D = sum(s -> size(s.values,1),spaces)
  values = zeros(TData,D,length(x),K)
  off = 0
  @inbounds @views for s in spaces
    xi = get_coords(s.space)
    pos = indexin(xi,x)
    d = size(s.values,1)
    values[off+1:off+d,pos,:] .= s.values
    off += d
  end
  OutputSpace(space,values)
end

function merge_spaces(spaces::OutputSpace...)
  D = size(first(spaces).values,1)
  @check all(size(s.values,1)==D for s in spaces) "All output spaces must have the same number of channels"
  space = MultiGraphInducedSpace(map(s -> s.space,spaces)...)
  N = num_coords(space)
  K = sum(s -> size(s.values,3),spaces)
  values = zeros(TData,D,N,K)
  off = 0
  @inbounds @views for (s,lcoord_to_coords) in zip(spaces,space.l_to_lcoord_to_coords)
    k = size(s.values,3)
    for (lc,c) in enumerate(lcoord_to_coords)
      values[:,c,off+1:off+k] .= s.values[:,lc,:]
    end
    off += k
  end
  OutputSpace(space,values)
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
  @eval $f(vals::AbstractInputData,space::AbstractGraphInducedSpace) = $f(vals,get_graph(space))
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
  graph = build_graph(coords;kwargs...)
  GraphInducedSpace(graph,coords)
end

get_graph(space::GraphInducedSpace) = space.graph
get_coords(space::GraphInducedSpace) = space.coords

struct MultiGraphInducedSpace{N,A<:GraphInducedSpace,B<:AbstractGraph,C<:AbstractVector{<:Point},D<:Table} <: AbstractGraphInducedSpace
  lspaces::NTuple{N,A}
  graph::B
  coords::C
  l_to_lcoord_to_coords::D
end

function MultiGraphInducedSpace(spaces...;kwargs...)
  lspaces = map(x -> GraphInducedSpace(x;kwargs...),spaces)
  lcoords = map(get_coords,lspaces)
  sets = map(Set,lcoords)
  coords = collect(∪(sets...))
  graph = build_graph(coords;kwargs...)
  l_to_lcoord_to_coords = _lcoord_to_coords_map(lcoords,coords)
  MultiGraphInducedSpace(lspaces,graph,coords,l_to_lcoord_to_coords)
end

get_coords(space::MultiGraphInducedSpace) = space.coords
get_graph(space::MultiGraphInducedSpace) = space.graph

Base.length(space::MultiGraphInducedSpace) = length(space.lspaces)
Base.getindex(space::MultiGraphInducedSpace,i::Int) = space.lspaces[i]
Base.iterate(space::MultiGraphInducedSpace,state...) = iterate(space.lspaces,state...)

# norms

function assemble_mass(f::FunctionSpace)
  dΩ = get_measure(f)
  assemble_form((u,v) -> ∫(u⋅v)dΩ,f)
end

function assemble_mass(f::FunctionSpace{<:MultiFieldFESpace})
  dΩ = get_measure(f)
  m(u,v) = sum(u[i]⋅v[i] for i in 1:num_fields(f.space))
  assemble_form((u,v) -> ∫(m(u,v))*dΩ,f)
end

function assemble_stiffness(f::FunctionSpace)
  dΩ = get_measure(f)
  assemble_form((u,v) -> ∫(∇(v)⊙∇(u))dΩ,f)
end

function assemble_stiffness(f::FunctionSpace{<:MultiFieldFESpace})
  dΩ = get_measure(f)
  a(u,v) = sum(∇(u[i])⊙∇(v[i]) for i in 1:num_fields(f.space))
  assemble_form((u,v) -> ∫(a(u,v))*dΩ,f)
end

function assemble_mass_stiffness(f::FunctionSpace)
  dΩ = get_measure(f)
  assemble_form((u,v) -> ∫(u⋅v + ∇(v)⊙∇(u))dΩ,f)
end

function assemble_mass_stiffness(f::FunctionSpace{<:MultiFieldFESpace})
  dΩ = get_measure(f)
  am(u,v) = sum(u[i]⋅v[i] + ∇(u[i])⊙∇(v[i]) for i in 1:num_fields(f.space))
  assemble_form((u,v) -> ∫(am(u,v))*dΩ,f)
end

function assemble_form(fun,f::FunctionSpace)
  mat = assemble_matrix(fun,f.space,f.space)
  msk = get_node_masks(f.space)
  return TData.(mat[msk,msk])
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
  a2b = zeros(Int,length(a))
  for (i,ai) in enumerate(a)
    a2b[i] = findfirst(x -> x == ai,b)
  end
  a2b
end

function _lcoord_to_coords_map(lcoords,coords)
  data = zeros(Int,length(lcoords)*length(coords))
  ptrs = ones(Int32,length(lcoords)+1)
  for (i,lc) in enumerate(lcoords)
    ptrs[i+1] = ptrs[i] + length(lc)
    @views data[ptrs[i]:(ptrs[i+1]-1)] .= _a_to_b(lc,coords)
  end
  Table(data,ptrs)
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
  x = get_coords(f.space)
  off = 0
  us = ntuple(length(f.space)) do i
    space_i = f.space[i]
    d = maximum(get_free_dof_to_comp(space_i))
    fi = OutputSpace(space_i,f.values)
    xi = get_coords(space_i)
    pos = indexin(xi,x)
    vi = view(data,off+1:off+d,pos,:)
    off += d
    _free_values(fi,vi)
  end
  reduce(vcat,us)
end

# agfem compatibility


function FESpaces.gather_free_and_dirichlet_values!(
  fmdof_to_val::AbstractVector{T},
  dmdof_to_val::AbstractVector{T},
  f::FESpaceWithLinearConstraints,
  cell_to_ludof_to_val
  ) where T

  fdof_to_val = Vector{T}(undef,num_free_dofs(f.space))
  ddof_to_val = Vector{T}(undef,num_dirichlet_dofs(f.space))
  gather_free_and_dirichlet_values!(fdof_to_val,ddof_to_val,f.space,cell_to_ludof_to_val)
  FESpaces._setup_mdof_to_val!(
    fmdof_to_val,
    dmdof_to_val,
    fdof_to_val,
    ddof_to_val,
    f.mDOF_to_DOF,
    f.n_fdofs,
    f.n_fmdofs)
  fmdof_to_val,dmdof_to_val
end

function get_mfdof_to_fdof(f::FESpaceWithLinearConstraints)
  T = eltype(f.mDOF_to_DOF)
  mfdof_to_fdof = zeros(T,num_free_dofs(f))
  for mDOF in 1:length(f.mDOF_to_DOF)
    DOF = f.mDOF_to_DOF[mDOF]
    mdof = FESpaces._DOF_to_dof(mDOF,f.n_fmdofs)
    dof = FESpaces._DOF_to_dof(DOF,f.n_fdofs)
    if mdof > 0
      @check dof > 0
      mfdof_to_fdof[mdof] = dof
    end
  end
  return mfdof_to_fdof
end

for F in (:get_free_dof_to_comp,:get_free_dof_to_point)
  @eval begin
    function $F(f::FESpaceWithLinearConstraints)
      mfdof_to_fdof = get_mfdof_to_fdof(f)
      fdof_to_q = $F(f.space)
      view(fdof_to_q,mfdof_to_fdof)
    end
  end
end