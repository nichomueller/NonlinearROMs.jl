# case 1: only 1 space
# case 2: multiple spaces, but all have the same mesh
# -> several cell fields, only one measure
# case 3: multiple spaces, each with a different mesh
# -> several cell fields, several measures

abstract type DiscreteSpace end

get_coords(f::DiscreteSpace) = @abstractmethod
num_coords(f::DiscreteSpace) = length(get_coords(f))

abstract type FunctionSpace <: DiscreteSpace end

abstract type GraphInducedSpace <: DiscreteSpace end

function interpolation_values(f::FunctionSpace,a)
  x = get_coords(f)
  lazy_map(a,x)
end

struct SimpleSpace{A,B} <: FunctionSpace
  space::A
  coords::B
end

function SimpleSpace(space)
  coords = get_node_coordinates(space)
  SimpleSpace(space,coords)
end

const SimpleFESpace{A<:FESpace} = SimpleSpace{A}

get_coords(f::SimpleSpace) = f.coords

function get_measure(f::SimpleFESpace;factor=2)
  trian = get_triangulation(f.space)
  degree = factor*get_polynomial_order(f.space)
  Measure(trian,degree)
end

abstract type SpaceStyle end
struct CommonDomain <: SpaceStyle end
struct MultiDomain <: SpaceStyle end

struct MultiSpace{A<:SpaceStyle,B<:AbstractVector{<:SimpleSpace},C<:AbstractVector,D<:Table,E<:AbstractVector} <: FunctionSpace
  style::A
  spaces::B
  coords::C
  lcoord_to_coords::D
  coord_to_lcoords::D
  multiplicity::E
end

function MultiSpace(spaces::AbstractVector{<:SimpleSpace})
  style,coords,lcoord_to_coords,coord_to_lcoords,multiplicity = _coords_info(spaces)
  MultiSpace(style,spaces,coords,lcoord_to_coords,coord_to_lcoords,multiplicity)
end

function MultiSpace(spaces::AbstractVector)
  MultiSpace(map(SimpleSpace,spaces))
end

get_coords(f::MultiSpace) = f.coords

# utils

function _a_to_b(a,b)
  a2b = zeros(Int,length(b))
  for (i,bi) in enumerate(b)
    a2b[i] = findfirst(x -> x == bi,a)
  end
  a2b
end

function build_lcoord_to_coords(lcoords,coords)
  data = zeros(Int,length(lcoords)*length(coords))
  ptrs = ones(Int32,length(lcoords)+1)
  for (i,lc) in enumerate(lcoords)
    ptrs[i+1] = ptrs[i] + length(lc)
    @views data[ptrs[i]:(ptrs[i+1]-1)] .= _a_to_b(lc,coords)
  end
  Table(data,ptrs)
end

function build_coord_to_lcoords(coords,lcoords)
  data = zeros(Int,sum(map(length,lcoords)))
  ptrs = ones(Int32,length(lcoords)+1)
  for (i,lc) in enumerate(lcoords)
    ptrs[i+1] = ptrs[i] + length(lc)
    @views data[ptrs[i]:(ptrs[i+1]-1)] .= _a_to_b(coords,lc)
  end
  Table(data,ptrs)
end

function build_multiplicity(coord_to_lcoords,ncoords)
  mult = zeros(Int,ncoords)
  for lc in coord_to_lcoords
    for g in lc
      mult[g] += 1
    end
  end
  mult
end

function _coords_info(spaces)
  _coords = map(get_coords,spaces)
  sets = map(Set,_coords)
  coords = ∪(sets...)
  style = isempty(∩(sets...)) ? MultiDomain() : CommonDomain()
  lcoord_to_coords = build_lcoord_to_coords(_coords,coords)
  coord_to_lcoords = build_coord_to_lcoords(coords,_coords)
  multiplicity = build_multiplicity(coord_to_lcoords,length(coords))
  style,coords,lcoord_to_coords,coord_to_lcoords,multiplicity
end

get_node_coordinates(V) = @abstractmethod

struct FENodeCoordinates{D,T,A<:AbstractVector{Point{D,T}},B<:Table,C<:AbstractVector{<:Integer}} <: AbstractVector{Point{D,T}}
  coords::A
  node_to_dofs::B
  dof_to_nodes::C
end

Base.size(f::FENodeCoordinates) = size(f.coords)
Base.getindex(f::FENodeCoordinates,i::Integer) = f.coords[i]

function get_node_coordinates(V::SingleFieldFESpace)
  d2c = get_free_dof_coordinates(V)
  d2n,n2d = get_dof_to_node_maps(V)
  reps = lazy_map(first,n2d) # one representative dof per node
  n2c = lazy_map(Reindex(d2c),reps)
  FENodeCoordinates(n2c,n2d,d2n)
end

function get_dof_to_node_maps(V::SingleFieldFESpace)
  cell_dof_basis = get_fe_dof_basis(V)
  cell_dof_bases = get_data(cell_dof_basis)
  @notimplementedif !(testitem(cell_dof_bases) isa LagrangianDofBasis)

  dof_to_nodes = zeros(Int32,num_free_dofs(V))

  cell_dof_to_lnode = lazy_map(get_dof_to_node,cell_dof_bases)
  cell_dof_ids = get_cell_dof_ids(V)
  lnode_cache = array_cache(cell_dof_to_lnode)
  dof_cache = array_cache(cell_dof_ids)
  nnodes = 0
  for cell in eachindex(cell_dof_ids)
    lnodes = getindex!(lnode_cache,cell_dof_to_lnode,cell)
    dofs = getindex!(dof_cache,cell_dof_ids,cell)
    cell_lnode_to_node = Dict{Int,Int32}()
    for i in eachindex(dofs)
      dof = dofs[i]
      (dof > 0 && dof_to_nodes[dof] != 0) && (cell_lnode_to_node[lnodes[i]] = dof_to_nodes[dof])
    end
    for i in eachindex(dofs)
      dof = dofs[i]
      (dof <= 0 || dof_to_nodes[dof] != 0) && continue
      ln = lnodes[i]
      node = get!(cell_lnode_to_node,ln) do
        nnodes += 1
        nnodes
      end
      dof_to_nodes[dof] = node
    end
  end
  
  node_to_dofs_ptrs = zeros(Int32,1+nnodes)
  for dof in eachindex(dof_to_nodes)
    node = dof_to_nodes[dof]
    node > 0 && (node_to_dofs_ptrs[1+node] += 1)
  end
  length_to_ptrs!(node_to_dofs_ptrs)

  node_to_dofs_data = zeros(Int,node_to_dofs_ptrs[end]-1)
  for dof in eachindex(dof_to_nodes)
    node = dof_to_nodes[dof]
    if node > 0
      p = node_to_dofs_ptrs[node]
      node_to_dofs_data[p] = dof
      node_to_dofs_ptrs[node] += 1
    end
  end
  rewind_ptrs!(node_to_dofs_ptrs)
  node_to_dofs = Table(node_to_dofs_data,node_to_dofs_ptrs)

  return dof_to_nodes,node_to_dofs
end