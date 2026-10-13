# case 1: only 1 space
# case 2: multiple spaces, but all have the same mesh
# -> several cell fields, only one measure
# case 3: multiple spaces, each with a different mesh
# -> several cell fields, several measures

abstract type FunctionSpace end

get_measure(f::FunctionSpace;kwargs...) = @abstractmethod
get_coords(f::FunctionSpace) = @abstractmethod
num_coords(f::FunctionSpace) = length(get_coords(f))
assemble_integral(a,f::FunctionSpace) = @abstractmethod

function CellData.integrate(a,f::FunctionSpace)
  dc = integrate(a,get_measure(f))
  assemble_integral(dc,f)
end

function interpolation_values(f::FunctionSpace,a::CellField)
  a(get_coords(f))
end

function build_graph(f::FunctionSpace,args...;kwargs...)
  build_graph(get_coords(f),args...;kwargs...)
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

get_coords(f::SimpleFESpace) = get_free_dof_coordinates(f.space)

function interpolation_values(f::SimpleFESpace,a::CellField)
  trian_a = get_triangulation(a)
  trian_f = get_triangulation(f.space)
  if CellData.is_change_possible(trian_a,trian_f)
    fv = zero_free_values(f.space)
    cv = FESpaces._cell_vals(f.space,a)
    gather_free_values!(fv,f.space,cv)
  else
    a(get_coords(f))
  end
end

function interpolation_values(f::FunctionSpace,a::FEFunction)
  f.space === get_fe_space(a) && return get_free_dof_values(a)
  trian_a = get_triangulation(a)
  trian_f = get_triangulation(f.space)
  if CellData.is_change_possible(trian_a,trian_f)
    fv = zero_free_values(f.space)
    cv = FESpaces._cell_vals(f.space,a)
    gather_free_values!(fv,f.space,cv)
  else
    a(get_coords(f))
  end
end

function assemble_integral(a,f::SimpleFESpace)
  vecdata = collect_cell_vector(a,f.space)
  assemble_vector(f.space,vecdata)
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

function CellData.integrate(a,f::MultiSpace{MultiDomain})
  sum(map(f -> integrate(a,f),f.spaces))
end

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