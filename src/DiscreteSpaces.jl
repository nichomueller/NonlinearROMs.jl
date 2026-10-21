# case 1: only 1 space
# case 2: multiple spaces, but all have the same mesh
# -> several cell fields, only one measure
# case 3: multiple spaces, each with a different mesh
# -> several cell fields, several measures

abstract type DiscreteSpace end

get_coords(f::DiscreteSpace) = @abstractmethod
num_coords(f::DiscreteSpace) = length(get_coords(f))

function interpolation_values(f::DiscreteSpace,a)
  x = get_coords(f)
  lazy_map(a,x)
end

function get_coordinates(f::SingleFieldFESpace)
  get_free_dof_coordinates(f)
end

function get_coordinates(f::MultiFieldFESpace)
  vcat(map(get_coordinates,f)...)
end

struct FunctionSpace{A,B} <: DiscreteSpace
  space::A
  coords::B
end

function FunctionSpace(space)
  scalar_space = scalarise(space)
  coords = get_coordinates(scalar_space)
  FunctionSpace(scalar_space,coords)
end

function FunctionSpace(space::MultiFieldFESpace)
  trian = get_triangulation(first(space))
  if !all(s -> trian === get_triangulation(s),space)
    return FunctionSpace(space...)
  end
  scalar_spaces = map(scalarise,space)
  scalar_space = MultiFieldFESpace(scalar_spaces) # forces consecutive style
  coords = get_coordinates(scalar_space)
  FunctionSpace(scalar_space,coords)
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
    struct GraphInducedSpace{A<:Graphs.AbstractGraph,B<:AbstractVector{<:Point}} <: AbstractGraphInducedSpace
      graph::A
      coords::B
    end
"""
struct GraphInducedSpace{A<:AbstractGraph,B<:AbstractVector{<:Point}} <: AbstractGraphInducedSpace
  graph::A
  coords::B
  function GraphInducedSpace(graph::A,coords::B) where {A<:Graphs.AbstractGraph,B<:AbstractVector{<:Point}}
    @check nv(graph) == length(coords) "GraphInducedSpace needs exactly one coordinate per vertex"
    new{A,B}(graph,coords)
  end
end

function GraphInducedSpace(space;kwargs...)
  coords = get_coords(space)
  g = build_graph(coords;kwargs...)
  GraphInducedSpace(g,coords)
end

get_graph(space::GraphInducedSpace) = space.graph
get_coords(space::GraphInducedSpace) = space.coords

struct MultiGraphInducedSpace{A<:AbstractGraph,B<:AbstractVector{<:Point},D<:Table} <: AbstractGraphInducedSpace
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
  GraphInducedSpace(graph,lgraphs,coords,node_to_lnode)
end

get_graph(space::MultiGraphInducedSpace) = space.graph
get_coords(space::MultiGraphInducedSpace) = space.coords

# utils

scalarise(V) = @notimplemented

function scalarise(U::TrialFESpace)
  V = scalarise(get_fe_space(U))
  cell_node_ids = get_cell_dof_ids(V)
  diri_node = _restrict_dirichlet(cell_node_ids,num_dirichlet_dofs(U))
  TrialFESpace(view(get_dirichlet_values(U),diri_node),V)
end

function scalarise(U::ParamTrialFESpace)
  V = scalarise(get_fe_space(U))
  cell_node_ids = get_cell_dof_ids(V)
  diri_node = _restrict_dirichlet(cell_node_ids,num_dirichlet_dofs(U))
  ParamTrialFESpace(get_param_entry(get_dirichlet_values(U),diri_node),V)
end

function scalarise(V::UnconstrainedFESpace)
  cell_dof_basis = get_fe_dof_basis(V)
  cell_dof_bases = get_data(cell_dof_basis)
  @notimplementedif !(testitem(cell_dof_bases) isa LagrangianDofBasis)

  cell_dof_to_lnode = lazy_map(get_dof_to_node,cell_dof_bases)
  cell_dof_to_lcomp = lazy_map(get_dof_to_comp,cell_dof_bases)
  cell_to_scalar_ldof = lazy_map((ln,lc) -> ln[findall(isone,lc)],cell_dof_to_lnode,cell_dof_to_lcomp)

  P = eltype(testitem(cell_dof_bases))
  D = num_components(P)
  @notimplementedif !(num_free_dofs(V) % D == 0 && num_dirichlet_dofs(V) % D == 0) 
  
  nfree = round(Int,num_free_dofs(V) ÷ D)
  ndiri = round(Int,num_dirichlet_dofs(V) ÷ D)
  cell_node_ids = _restrict_dirichlet(get_cell_dof_ids(V),cell_to_scalar_ldof)
  cell_node_basis = _restrict(get_fe_dof_basis(V),cell_to_scalar_ldof)
  diri_node = _restrict_dirichlet(cell_node_ids,num_dirichlet_dofs(V))
  diri_node_tag = get_dirichlet_dof_tag(V)[diri_node]
  UnconstrainedFESpace(
    get_vector_type(V),
    nfree,
    ndiri,
    cell_node_ids,
    get_fe_basis(V),
    cell_node_basis,
    get_cell_is_dirichlet(V),
    diri_node_tag,
    get_dirichlet_cells(V),
    V.ntags
  )
end

_restrict(a,ldof_map) = lazy_map((x,ldof) -> x[ldof],a,ldof_map)

function _restrict(a::CellDof,ldof_map)
  CellDof(_restrict(get_data(a),ldof_map),get_triangulation(a),DomainStyle(a))
end

function _restrict_dirichlet(cell_node_ids,ndiri)
  diri_nodes = falses(ndiri)
  cache = array_cache(cell_node_ids)
  for cell in eachindex(cell_node_ids)
    nodes = getindex!(cache,cell_node_ids,cell)
    for node in nodes
      if node < 0
        diri_nodes[abs(node)] = 1
      end
    end
  end
  findall(diri_nodes)
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