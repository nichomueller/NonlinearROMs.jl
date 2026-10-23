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

function get_coords(f::DiscreteSpace,a::FeatureBundle)
  x = get_coords(f)
  D = num_components(eltype(x))
  N = length(x)
  K = num_params(a)
  coords = zeros(TData,D,N,K)
  for (i,xi) in enumerate(x)
    for d in 1:D 
      @views coords[d,i,:] .= xi.data[d]
    end
  end
  return coords
end

function get_values(a::FeatureBundle,f::DiscreteSpace)
  features,offsets = _init_data(f,a)
  @inbounds for (j,a) in enumerate(a.bundle)
    af = interpolation_values(f,a)
    _set_data!(features,af,offsets[j])
  end
  return features
end

function get_values(uh::FEFunction,f::DiscreteSpace)
  values, = _init_data(f,a)
  u = interpolation_values(f,uh)
  _set_data!(values,u,0)
  return values
end

struct DataFormatter{A,B,C,D}
  pass_coords::Val{A}
  pass_features::Val{B}
  pass_new_data::Val{C}
  sampler::Sampler{D}
end

function DataFormatter(;
  pass_coords=Val{true}(),
  pass_features=Val{true}(),
  pass_new_data=Val{false}(),
  sampler=Sampler()
  )

  DataFormatter(pass_coords,pass_features,pass_new_data,sampler)
end

function _fetch_data(d::DataFormatter{false,false},f::DiscreteSpace,b::FeatureBundle)
  (coords = nothing,features = nothing)
end

function _fetch_data(d::DataFormatter{true,false},f::DiscreteSpace,b::FeatureBundle)
  coords = sample(d.sampler,get_coords(f,b))
  (coords = coords,features = nothing)
end

function _fetch_data(d::DataFormatter{false,true},f::DiscreteSpace,b::FeatureBundle)
  feats = sample(d.sampler,get_values(f,b))
  (coords = nothing,features = feats)
end

function _fetch_data(d::DataFormatter{true,true},f::DiscreteSpace,b::FeatureBundle)
  coords = sample(d.sampler,get_coords(f,b))
  features = sample(d.sampler,get_values(f,b))
  (coords = coords,features = features)
end

function _fetch_data(d::DataFormatter,f::DiscreteSpace,uh::ParamFEFunction)
  sample(d.sampler,get_values(f,uh))
end

_merge_data(a,b) = @abstractmethod
_merge_data(a::Nothing,b::Nothing) = nothing
_merge_data(a::Nothing,b) = b
_merge_data(a,b::Nothing) = a
_merge_data(a::InputData,b::InputData) = cat(a,b;dims=1)

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

function fetch_data(
  d::DataFormatter,
  f::DiscreteSpace,
  b::FeatureBundle
  )
  
  (input = _fetch_data(d,f,b),)
end

function fetch_data(
  d::DataFormatter,
  f::DiscreteSpace,
  b::FeatureBundle,
  uh::ParamFEFunction
  )
  
  (input = _fetch_data(d,f,b),output = sample(d.sampler,get_values(f,uh)))
end

function fetch_data_and_stats(d::DataFormatter,args...)
  data = fetch_data(args...)
  stats = Normalisation(data;normalise=d.normalise)
  mdata = _merge_data(data)
  return mdata,stats
end

function fetch_data_from_stats(stats::Normalisation,d::DataFormatter,args...)
  data = fetch_data(args...)
  normalise!(data,stats;normalise=d.normalise)
  mdata = _merge_data(data)
  return mdata
end

# utils

function _init_data(f::DiscreteSpace,a::FeatureBundle)
  x = get_coords(f)
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

# data formatter:
# 1) x
# 2) x, (a(x),...,f(x))
# 3) x, vₜ(x)
# 4) x, (a(x),...,f(x)), vₜ(x)
# 5) DeepONet, NOMAD: branch: μ, trunk: x