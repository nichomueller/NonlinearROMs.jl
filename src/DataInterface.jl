struct CoordsSampler{A<:Union{AbstractVector,Colon}}
  strategy::A
end

CoordsSampler() = CoordsSampler(:)

function sample_coords(s::CoordsSampler,f::DiscreteSpace)
  x = get_coords(f)
  view(x,s.strategy)
end

struct FeatureBundle{N,T<:AbstractParamFunction}
  bundle::NTuple{N,T}
  function FeatureBundle(bundle::NTuple{N,T}) where {N,T<:AbstractParamFunction}
    r = get_realisation(first(bundle))
    @check all(get_realisation(a)==r for a in bundle) "All param functions must have the same realisation"
    new{N,T}(bundle)
  end
end

const TransientFeatureBundle{N} = FeatureBundle{N,<:TransientParamFunction}

ParamDataStructures.get_realisation(a::FeatureBundle) = get_realisation(first(a.bundle))
ParamDataStructures.get_params(a::FeatureBundle) = get_params(get_realisation(a))
ParamDataStructures.get_times(a::FeatureBundle) = get_times(get_realisation(a))
ParamDataStructures.num_params(a::FeatureBundle) = num_params(get_realisation(a))
ParamDataStructures.num_times(a::FeatureBundle) = num_times(get_realisation(a))

function get_features(a::FeatureBundle,f::DiscreteSpace)
  features = init_features(f,a)
  @inbounds for (j,a) in enumerate(a.bundle)
    af = interpolation_values(f,a)
    set_features!(features,af,j)
  end
  return features
end

function sample_features(s::CoordsSampler,args...)
  features = get_features(args...)
  c = ntuple(i->Colon(),Val{ndims(features)-1}())
  view(features,s.strategy,c...)
end

struct DataFormatter{A,B,C}
  pass_coords::Val{A}
  sampler::CoordsSampler{B}
  metadata::C
end

const BundleDataFormatter{A,B} = DataFormatter{A,B,<:FeatureBundle}

function DataFormatter(pass_coords::Val,sampler::CoordsSampler)
  DataFormatter(pass_coords,sampler,nothing)
end

function DataFormatter(args...;pass_coords=Val{true}(),sampler=CoordsSampler())
  DataFormatter(pass_coords,sampler,args...)
end

function _fetch_data(d::DataFormatter{false},f::DiscreteSpace)
  nothing
end

function _fetch_data(d::DataFormatter{true},f::DiscreteSpace)
  x = sample_coords(d.sampler,f)
  matrix_of_coords(x)
end

function _fetch_data(d::BundleDataFormatter{false},f::DiscreteSpace)
  sample_features(d.sampler,d.metadata,f)
end

function _fetch_data(d::BundleDataFormatter{true},f::DiscreteSpace)
  x = sample_coords(d.sampler,f)
  mat = matrix_of_coords(x)
  features = sample_features(d.sampler,d.metadata,f)
  stack((mat,features),dims=1)
end

function fetch_data(d::DataFormatter,f::DiscreteSpace)
  _fetch_data(d,f)
end

function fetch_data(d::DataFormatter,f::AbstractGraphInducedSpace)
  data = _fetch_data(d,f)
  fetch_data(data,get_graph(f))
end

struct LayerData{A,B}
  pass_features::Val{A}
  fixed_data::B
end

function LayerData(fixed_data;pass_features=Val{true}())
  LayerData(pass_features,fixed_data)
end

fetch_data(d::LayerData{false},x) = d.fixed_data
fetch_data(d::LayerData{true},x) = stack((d.fixed_data,x),dims=1)

# utils

function init_features(f::DiscreteSpace,a::FeatureBundle)
  x = get_coords(f)
  d = _dimension(a,testitem(x))
  N = length(x)
  K = num_params(a)
  zeros(Float32,d,N,K)
end

function init_features(f::DiscreteSpace,a::TransientFeatureBundle)
  x = get_coords(f)
  d = _dimension(a,testitem(x))
  N = length(x)
  K = num_params(a)
  T = num_times(a)
  feats = zeros(Float32,d,N,K)
  RolloutData(feats,T)
end

function _dimension(a::FeatureBundle,x::Point)
  d = 0
  for f in a.bundle
    T = return_type(f,x)
    d += num_components(eltype(T))
  end
  return d
end

function set_features!(cache,interp,j)
  @views cache[:,j] .= get_interp_values(interp)
end

function set_features!(cache,interp::ConsecutiveParamVector,j)
  @views cache[:,j,:] .= get_interp_values(interp)
end

# data formatter:
# 1) x
# 2) x, (a(x),...,f(x))
# 3) x, vₜ(x)
# 4) x, (a(x),...,f(x)), vₜ(x)
# 5) DeepONet, NOMAD: branch: μ, trunk: x