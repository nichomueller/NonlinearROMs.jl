struct Sampler{A<:Union{AbstractVector,Colon}}
  strategy::A
end

Sampler() = Sampler(:)

function sample_coords(s::Sampler,f::DiscreteSpace)
  x = get_coords(f)
  mat = matrix_of_coords(x)
  view(mat,:,s.strategy)
end

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

function get_features(a::FeatureBundle,f::DiscreteSpace)
  features,offsets = init_features(f,a)
  @inbounds for (j,a) in enumerate(a.bundle)
    af = interpolation_values(f,a)
    set_features!(features,af,offsets[j])
  end
  return features
end

function sample_features(s::Sampler,a::FeatureBundle,f::DiscreteSpace)
  features = get_features(a,f)
  view(features,:,s.strategy,:)
end

function get_values(uh::FEFunction,f::DiscreteSpace)
  values = init_features(f,a)
  u = interpolation_values(f,uh)
  set_features!(values,u,0)
  return values
end

function sample_values(s::Sampler,uh::FEFunction,f::DiscreteSpace)
  u = get_values(f,uh)
  view(u,:,s.strategy,:)
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
  x = sample_coords(d.sampler,f)
  (coords = x,features = nothing)
end

function _fetch_data(d::DataFormatter{false,true},f::DiscreteSpace,b::FeatureBundle)
  feats = sample_features(d.sampler,b,f)
  (coords = nothing,features = feats)
end

function _fetch_data(d::DataFormatter{true,true},f::DiscreteSpace,b::FeatureBundle)
  x = sample_coords(d.sampler,f)
  features = sample_features(d.sampler,b,f)
  (coords = x,features = features)
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
  uh::FEFunction
  )
  
  (input = _fetch_data(d,f,b),output = sample_values(d.sampler,uh,f))
end

function fetch_data_and_stats(args...;kwargs...)
  data = fetch_data(args...)
  stats = Normalisation(data;kwargs...)
  return data,stats
end

function fetch_data_and_stats(stats::Normalisation,args...;kwargs...)
  data = fetch_data(args...)
  normalise!(data,stats)
  return data,stats
end

struct LayerData{A,B}
  pass_new_data::Val{A}
  fixed_data::B
end

function LayerData(fixed_data;pass_new_data=Val{true}())
  LayerData(pass_new_data,fixed_data)
end

fetch_data(d::LayerData{false},x) = d.fixed_data
fetch_data(d::LayerData{true},x) = cat(d.fixed_data,x,dims=1)

# utils

function init_features(f::DiscreteSpace,a::FeatureBundle)
  x = get_coords(f)
  offsets = _offsets(a,testitem(x))
  D = offsets[end]
  N = length(x)
  K = num_params(a)
  feats = zeros(Float32,D,N,K)
  return feats,offsets
end

function _offsets(a::FeatureBundle,x::Point)
  offsets = zeros(Int,length(a.bundle)+1)
  for (i,f) in 1:length(a.bundle)
    T = return_type(f,x)
    offsets[i+1] = offsets[i] + num_components(eltype(T))
  end
  return offsets
end

function set_features!(feats,interp,o)
  cache = array_cache(interp)
  @inbounds for i in eachindex(interp)
    vals = getindex!(cache,interp,i)
    for k in param_eachindex(vals)
      _set_feature!(feats,vals[k],o,i,k)
    end
  end
  feats
end

_set_feature!(feats,vals,o,i,k) = (feats[o+1,i,k] = vals)

function _set_feature!(feats,vals::Point{D},o,i,k) where D
  for d in 1:D
    feats[o+d,i,k] = vals.data[d]
  end
end

# data formatter:
# 1) x
# 2) x, (a(x),...,f(x))
# 3) x, vₜ(x)
# 4) x, (a(x),...,f(x)), vₜ(x)
# 5) DeepONet, NOMAD: branch: μ, trunk: x