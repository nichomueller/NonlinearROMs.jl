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

struct SimpleSpace{A} <: FunctionSpace
  space::A
end

const SimpleFESpace{A<:FESpace} = SimpleSpace{A}

function get_measure(f::SimpleFESpace;factor=2)
  trian = get_triangulation(f.space)
  degree = factor*get_polynomial_order(f.space)
  Measure(trian,degree)
end

get_coords(f::SimpleFESpace) = get_node_coordinates(f.space)

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

function get_node_coordinates(V::SingleFieldFESpace)
  cell_dof_basis = get_fe_dof_basis(V)
  cell_points = change_domain(get_cell_points(cell_dof_basis),PhysicalDomain())
  cell_phys_nodes = get_data(cell_points)
  get_nodal_values(V,cell_phys_nodes)
end

function get_value_coordinates(uh::FEFunction)
  V = get_fe_space(uh)
  cell_dof_values = get_cell_dof_values(uh)
  cell_values = 
  get_nodal_values(V,cell_values)
end

function get_nodal_values(V::SingleFieldFESpace,cell_values)
  cell_dof_basis = get_fe_dof_basis(V)
  cell_dof_bases = get_data(cell_dof_basis)
  @notimplementedif !(testitem(cell_dof_bases) isa LagrangianDofBasis)

  cell_dof_to_node = lazy_map(get_dof_to_node,cell_dof_bases)
  cell_dof_to_comp = lazy_map(get_dof_to_comp,cell_dof_bases)
  cell_node_to_node = lazy_map((d2n,d2c) -> d2n[findall(d2c .== 1)],cell_dof_to_node,cell_dof_to_comp)

  v = testitem(cell_values)
  P = eltype(v)
  T = eltype(P)
  D = num_components(P)
  
  nnodes = Int(num_free_dofs(V) ÷ D)
  nsamples = num_samples(v)
  free_vals = Array{T,3}(undef,D,nnodes,nsamples)

  gather_nodal_values!(free_vals,V,cell_values,cell_node_to_node)
  return free_vals
end

function gather_nodal_values!(free_vals,V::SingleFieldFESpace,cell_vals,cell_node_to_node)
  cell_dofs = get_cell_dof_ids(V)
  cache_vals = array_cache(cell_vals)
  cache_dofs = array_cache(cell_dofs)
  cache_nodes = array_cache(cell_node_to_node)
  cells = 1:length(cell_vals)

  for cell in cells
    vals = getindex!(cache_vals,cell_vals,cell)
    dofs = getindex!(cache_dofs,cell_dofs,cell)
    nodes = getindex!(cache_nodes,cell_node_to_node,cell)
    for i in nodes
      val = vals[i]
      dof = dofs[i]
      if dof > 0
        for d in eachindex(val)
          free_vals[d,dof,1] = val[d]
        end
      end
    end
  end
end

function gather_nodal_values!(free_vals,V::SingleFieldFESpace,cell_vals::AbstractArray{<:ParamBlock},cell_node_to_node)
  cell_dofs = get_cell_dof_ids(V)
  cache_vals = array_cache(cell_vals)
  cache_dofs = array_cache(cell_dofs)
  cache_nodes = array_cache(cell_node_to_node)
  cells = 1:length(cell_vals)

  for cell in cells
    vals = getindex!(cache_vals,cell_vals,cell)
    dofs = getindex!(cache_dofs,cell_dofs,cell)
    nodes = getindex!(cache_nodes,cell_node_to_node,cell)
    for i in nodes
      val = vals[i]
      dof = dofs[i]
      if dof > 0
        for d in eachindex(val)
          for k in num_samples(val)
            free_vals[d,dof,k] = get_all_data(val)[d,k]
          end
        end
      end
    end
  end
end

num_samples(v) = 1
num_samples(v::Union{ParamBlock,AbstractParamData}) = param_length(v)