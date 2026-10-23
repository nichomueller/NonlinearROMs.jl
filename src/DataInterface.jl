struct Sampler{A<:Union{AbstractVector,Colon}}
  strategy::A
end

Sampler() = Sampler(:)

sample(s::Sampler,data::InputData) = view(data,:,s.strategy,:)

struct FeatureBundle{N,T<:AbstractParamFunction}
  bundle::NTuple{N,T}
  function FeatureBundle(bundle::NTuple{N,T}) where {N,T<:AbstractParamFunction}
    r = get_realisation(first(bundle))
    @check all(get_realisation(a)==r for a in bundle) "All param functions must have the same realisation"
    new{N,T}(bundle)
  end
end

ParamDataStructures.get_realisation(a::FeatureBundle) = get_realisation(first(a.bundle))
ParamDataStructures.get_params(a::FeatureBundle) = get_params(get_realisation(a))
ParamDataStructures.get_times(a::FeatureBundle) = get_times(get_realisation(a))
ParamDataStructures.num_params(a::FeatureBundle) = num_params(get_realisation(a))
ParamDataStructures.num_times(a::FeatureBundle) = num_times(get_realisation(a))

get_param_fun(a::GenericField) = a.object
get_param_fun(a::ParamFEFunction) = get_param_fun(testitem(get_data(get_data(a))))
ParamDataStructures.get_params(a::FeatureBundle) = get_params(get_param_fun(a))
ParamDataStructures.get_times(a::FeatureBundle) = get_times(get_param_fun(a))
ParamDataStructures.num_params(a::FeatureBundle) = num_params(get_params(a))
ParamDataStructures.num_times(a::FeatureBundle) = length(get_times(a))

struct DataFormatter{A,B,C,D}
  pass_coords::Val{A}
  pass_features::Val{B}
  pass_new_data::Val{C}
  normalise::Bool
  sampler::Sampler{D}
end

function DataFormatter(;
  pass_coords=Val{true}(),
  pass_features=Val{true}(),
  pass_new_data=Val{false}(),
  normalise=true,
  sampler=Sampler()
  )

  DataFormatter(pass_coords,pass_features,pass_new_data,normalise,sampler)
end

# input
function fetch_data(::DataFormatter{false,false},(coords,features))
  @notimplemented "Wait ... no input?"
end

# input
function fetch_data(::DataFormatter{true,false},(coords,features))
  (
    coords = sample(d.sampler,coords),
    features = nothing
  )
end

# input
function fetch_data(d::DataFormatter{false,true},(coords,features))
  (
    coords = nothing,
    features = sample(d.sampler,features)
  )
end

# input
function fetch_data(d::DataFormatter{true,true},(coords,features))
  (
    coords = sample(d.sampler,coords),
    features = sample(d.sampler,features)
  )
end

# output 
function fetch_data(d::DataFormatter,values::InputData)
  sample(d.sampler,values)
end

function fetch_stats(d::DataFormatter,data)
  Normalisation(data;normalise=d.normalise)
end

function merge_data(::DataFormatter,data::NamedTuple)
  @check haskey(data,:input)
  input = _merge_data(data.input.coords,data.input.features)
  if haskey(data,:output)
    output = data.output
    return (input=input,output=output)
  else
    return (input=input,)
  end
end

function merge_data(::DataFormatter{<:Any,<:Any,false},fixed_data,new_data)
  fixed_data
end

function merge_data(::DataFormatter{<:Any,<:Any,true},fixed_data,new_data)
  _merge_data(fixed_data,new_data)
end

# utils

function _get_coords(x::AbstractVector{<:Point},K::Int)
  D = num_components(eltype(x))
  N = length(x)
  coords = zeros(TData,D,N,K)
  for (i,xi) in enumerate(x)
    for d in 1:D 
      @views coords[d,i,:] .= xi.data[d]
    end
  end
  return coords
end

function _get_features(a::FeatureBundle,x::AbstractVector{<:Point})
  features,offsets = _init_data(a,x)
  @inbounds for (j,a) in enumerate(a)
    af = lazy_map(a,x)
    _set_data!(features,af,offsets[j])
  end
  return features
end

function _get_values(ah::ParamFEFunction,x::AbstractVector{<:Point})
  values,offsets = _init_data(ah,x)
  a = lazy_map(ah,x)
  _set_data!(values,a,first(offsets))
  return values
end

function _init_data(a,x)
  offsets = _offsets(a,testitem(x))
  D = offsets[end]
  N = length(x)
  K = num_params(a)
  data = zeros(TData,D,N,K)
  return data,offsets
end

function _offsets(a::FeatureBundle,x::Point)
  offsets = zeros(Int,length(a.bundle)+1)
  for (i,f) in 1:length(a.bundle)
    T = return_type(f,x)
    offsets[i+1] = offsets[i] + num_components(eltype(T))
  end
  return offsets
end

function _offsets(a::ParamFEFunction,x::Point)
  offsets = zeros(Int,2)
  T = return_type(a,x)
  offsets[2] = num_components(eltype(T))
  return offsets
end

function _set_data!(data,vals,o)
  cache = array_cache(vals)
  @inbounds for i in eachindex(vals)
    valsi = getindex!(cache,vals,i)
    for k in param_eachindex(valsi)
      _set_data!(data,param_getindex(valsi,k),o,i,k)
    end
  end
  data
end

_set_data!(data,valsik,o,i,k) = (data[o+1,i,k] = valsik)

function _set_data!(data,valsik::Point{D},o,i,k) where D
  for d in 1:D
    data[o+d,i,k] = valsik.data[d]
  end
end

_merge_data(a,b) = @abstractmethod
_merge_data(a::Nothing,b::Nothing) = nothing
_merge_data(a::Nothing,b) = b
_merge_data(a,b::Nothing) = a
_merge_data(a::InputData,b::InputData) = cat(a,b;dims=1)

# data formatter:
# 1) x
# 2) x, (a(x),...,f(x))
# 3) x, vₜ(x)
# 4) x, (a(x),...,f(x)), vₜ(x)
# 5) DeepONet, NOMAD: branch: μ, trunk: x