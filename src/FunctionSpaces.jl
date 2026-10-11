struct FreeForm
  # fields: do this
end

function CellData.integrate(a,ff::FreeForm)
  @notimplemented "Do this"
end

abstract type FunctionSpace end

get_coords(fs::FunctionSpace) = @abstractmethod
get_measure(fs::FunctionSpace) = @abstractmethod

function CellData.integrate(a,fs::FunctionSpace)
  measure = get_measure(fs)
  return integrate(a,measure)
end

function find_common_space(a::FunctionSpace,b::FunctionSpace)
  @notimplemented "The two spaces should be of the same type, detected $(typeof(a)) != $(typeof(b))"
end

function find_common_space(a::A,b::A) where A<:FunctionSpace
  @notimplemented "Do this"
end

function find_common_space(a...)
  b,c,d... = a
  find_common_space(find_common_space(b,c),d...)
end

struct GenericFunctionSpace{A,B} <: FunctionSpace
  basis::A
  measure::B
end

function FunctionSpace(args...)
  GenericFunctionSpace(FreeForm(args...))
end

function FESpaces.interpolate(a,fs::GenericFunctionSpace)
  @notimplemented "Do this"
end

struct FunctionFESpace{A<:FESpace} <: FunctionSpace
  space::A
end

function FunctionSpace(space::FESpace)
  FunctionFESpace(space)
end

function get_coords(fs::FunctionFESpace)
  get_free_dof_coordinates(fs.space)
end

function get_measure(fs::FunctionFESpace)
  trian = get_triangulation(fs.space)
  order = get_polynomial_order(fs.space)
  Measure(trian,2*order)
end

function FESpaces.interpolate(a,fs::FunctionFESpace)
  interpolate(a,fs.space)
end

abstract type Cloud end

get_space(a) = @abstractmethod

get_coords(a::Cloud) = get_coords(get_space(a))
get_measure(a::Cloud) = get_measure(get_space(a))

struct GenericCloud{A<:FunctionSpace,B<:AbstractArray} <: Cloud
  space::A
  values::B
end

get_space(a::GenericCloud) = a.space

struct FECloud{A<:FEFunction} <: Cloud
  fh::A
end

get_space(a::FECloud) = get_fe_space(a.fh)

struct SeededCloud{A<:Cloud,B<:AbstractRealisation} <: Cloud
  fh::A
  seed::B
end

const TransientSeededCloud{A<:Cloud} = SeededSeededCloud{A,<:TransientRealisation} where A<:Cloud

get_space(a::SeededCloud) = evaluate(get_space(a.fh),a.seed)

# bundles 

struct FeaturesBundle{A<:Tuple}
  bundle::A
end

FeaturesBundle(args::Function...) = FeaturesBundle(args)

function extract_features(a::FeaturesBundle,fs::FunctionSpace)
  features = zeros(length(get_coords(fs)),length(a.bundle))
  @inbounds @views for (j,a) in enumerate(a.bundle)
    ah = interpolate(a,fs)
    fv = get_interp_values(ah)
    features[:,j] = fv
  end
  return features
end

function extract_features(f::FeaturesBundle,a::Cloud)
  extract_features(f,get_space(a))
end

function get_features(f::FeaturesBundle,a::SeededCloud)
  space = get_space(a)
  features = zeros(length(get_coords(a)),length(f.bundle),num_params(a.seed))
  @inbounds @views for (j,f) in enumerate(f.bundle)
    fh = interpolate(f(a.seed),space)
    fv = get_interp_values(fh)
    for k in param_eachindex(fv)
      features[:,j,k] = param_getindex(fv,k)
    end
  end
  return features
end

function get_features(f::FeaturesBundle,a::TransientSeededCloud)
  space = get_space(a)
  features = zeros(length(get_coords(a)),num_times(a.seed),length(f.bundle),num_params(a.seed))
  @inbounds @views for (j,f) in enumerate(f.bundle)
    fh = interpolate(f(a.seed),space)
    fv = get_interp_values(fh)
    for k in param_eachindex(fv)
      kt = slow_index(k,num_params(a.seed))
      kp = fast_index(k,num_params(a.seed))
      features[:,kt,j,kp] = param_getindex(fv,k)
    end
  end
  return features
end

function ()
  
end

# utils

function find_common_space(a::A,b::A) where A<:SingleFieldFESpace
  a === b && return a
  trian_a = get_triangulation(a)
  trian_b = get_triangulation(b)
  if trian_a === trian_b
    if num_free_dofs(a) ≥ num_free_dofs(b)
      return a
    else
      return b
    end
  else
    @notimplemented "Do this"
  end
end

get_interp_values(a) = @abstractmethod
get_interp_values(a::FEFunction) = get_free_dof_values(a)