# case 1: only 1 space
# case 2: multiple spaces, but all have the same mesh
# -> several cell fields, only one measure
# case 3: multiple spaces, each with a different mesh
# -> several cell fields, several measures

abstract type FunctionSpace end

get_measure(f::FunctionSpace;kwargs...) = @abstractmethod
get_coords(f::FunctionSpace) = @abstractmethod

function CellData.integrate(a,f::FunctionSpace)
  dc = integrate(a,get_measure(f))
  sum(dc)
end

function interpolation_values(f::FunctionSpace,a::CellField)
  a(get_coords(f))
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

abstract type SpaceStyle end
struct CommonDomain <: SpaceStyle end
struct MultiDomain <: SpaceStyle end

function same_domain(a,b)
  @notimplemented "Do this"
end

function same_domain(a::FESpace,b::FESpace)
  trian_a = get_triangulation(a)
  trian_b = get_triangulation(b)
  CellData.is_change_possible(trian_a,trian_b)
end

function same_domain(a::SimpleSpace,b::SimpleSpace)
  same_domain(a.space,b.space)
end

function same_domain(a,b...)
  c,d... = b
  if same_domain(a,c)
    same_domain(c,d...)
  else
    return false
  end
end

function build_coords(::CommonDomain,spaces::AbstractVector{<:FESpace})
  f = first(spaces)
  get_free_dof_coordinates(f)
end

function build_coords(::MultiDomain,spaces::AbstractVector{<:FESpace})
  coords = map(f -> Set(get_free_dof_coordinates(f)),spaces)
  ∪(coords...)
end

struct MultiSpace{A<:SpaceStyle,B<:AbstractVector{<:SimpleSpace},C<:AbstractVector{<:Point}} <: FunctionSpace
  style::A
  spaces::B
  coords::C
end

function MultiSpace(style::SpaceStyle,spaces::AbstractVector{<:SimpleSpace})
  coords = build_coords(style,spaces)
  MultiSpace(style,spaces,coords)
end

function MultiSpace(spaces::AbstractVector{<:SimpleSpace})
  style = same_domain(spaces...) ? CommonDomain() : MultiDomain()
  MultiSpace(style,spaces)
end

function MultiSpace(spaces::AbstractVector{<:FESpace})
  MultiSpace(map(SimpleSpace,spaces))
end

function get_measure(f::MultiSpace{CommonDomain};kwargs...)
  get_measure(first(f.spaces);kwargs...)
end

get_coords(f::MultiSpace) = f.coords

function CellData.integrate(a,f::MultiSpace{MultiDomain})
  sum(map(f -> integrate(a,f),f.spaces))
end