struct CoordsSampler{A<:Union{AbstractVector,Colon}}
  strategy::A
end

CoordsSampler() = CoordsSampler(:)

function sample_coords(s::CoordsSampler,f::DiscreteSpace)
  x = get_coords(f)
  view(x,s.strategy)
end

struct FeatureBundle{A<:Tuple}
  bundle::A
end

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

function fetch_data(d::DataFormatter{false},f::DiscreteSpace)
  nothing
end

function fetch_data(d::DataFormatter{true},f::DiscreteSpace)
  x = sample_coords(d.sampler,f)
  matrix_of_coords(x)
end

function fetch_data(d::BundleDataFormatter{false},f::DiscreteSpace)
  sample_features(d.sampler,d.metadata,f)
end

function fetch_data(d::BundleDataFormatter{true},f::DiscreteSpace)
  x = sample_coords(d.sampler,f)
  mat = matrix_of_coords(x)
  features = sample_features(d.sampler,d.metadata,f)
  stack((mat,features),dims=1)
end

function gather_data(d::DataFormatter,f::DiscreteSpace)
  fetch_data(d,f)
end

function gather_data(d::DataFormatter,f::GraphInducedSpace)
  data = fetch_data(d,f)
  gather_data(data,get_space(f))
end

struct LayerData{A,B}
  pass_features::Val{A}
  fixed_data::B
end

function LayerData(fixed_data;pass_features=Val{true}())
  LayerData(pass_features,fixed_data)
end

gather_data(d::LayerData{false},x) = d.fixed_data
gather_data(d::LayerData{true},x) = stack((d.fixed_data,x),dims=1)

# utils

_underlying_space(f::SimpleSpace) = f.space
_underlying_space(f::MultiSpace) = _underlying_space(first(f.space))

function init_features(f,a)
  T = eltype2(get_vector_type(f))
  s = _features_size(_underlying_space(f),f,a)
  zeros(T,s...)
end

function _features_size(uf,f,a)
  @abstractmethod
end

function _features_size(uf::FESpace,f,a)
  nx = num_coords(f)
  nb = length(a.bundle)
  (nx,nb)
end

function _features_size(uf::SingleFieldParamFESpace,f,a)
  nx = num_coords(f)
  nb = length(a.bundle)
  np = param_length(uf)
  (nx,nb,np)
end

function set_features!(cache,interp,j)
  @views cache[:,j] .= get_interp_values(interp)
end

function set_features!(cache,interp::ConsecutiveParamVector,j)
  @views cache[:,j,:] .= get_interp_values(interp)
end

#TODO missing interface for transient param functions

# data formatter:
# 1) x
# 2) x, (a(x),...,f(x))
# 3) x, vₜ(x)
# 4) x, (a(x),...,f(x)), vₜ(x)
# 5) DeepONet, NOMAD: branch: μ, trunk: x