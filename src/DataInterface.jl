struct Sampler{A<:Union{AbstractVector,Colon}}
  strategy::A
end

Sampler() = Sampler(:)

sample(s::Sampler,data::AbstractInputData) = view(data,:,s.strategy,:)

struct FeatureBundle{N,T<:NTuple{N,AbstractParamFunction}}
  bundle::T
end

function merge_bundles(a::FeatureBundle,b::FeatureBundle)
  FeatureBundle((a.bundle...,b.bundle...))
end

function merge_bundles(a::FeatureBundle,b::FeatureBundle,args::FeatureBundle...)
  merge_bundles(merge_bundles(a,b),args...)
end

function ParamDataStructures.num_params(a::FeatureBundle)
  dict = OrderedDict{UInt64,AbstractRealisation}()
  for f in a.bundle
    μ = get_params(f)
    dict[objectid(μ)] = μ
  end
  np = 0
  for μ in values(dict)
    np += num_params(μ)
  end
  return np
end

ParamDataStructures.num_params(a::ParamFEFunction) = param_length(a)

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

# Constructor for the 3 formulation of Kovachki et al. (Sez. 3, Eq. 7, 8, 9):
# Eq. (7): κ(x, y)
SpatialKernelFormatter(;kwargs...) = DataFormatter(;pass_coords=Val(true),pass_features=Val(false),pass_new_data=Val(false),kwargs...)
# Eq. (8): κ(x, y, a(x), a(y))
FeatureKernelFormatter(;kwargs...) = DataFormatter(;pass_coords=Val(true),pass_features=Val(true),pass_new_data=Val(false),kwargs...)
# Eq. (9): κ(x, y, v(x), v(y))
StateKernelFormatter(;kwargs...) = DataFormatter(;pass_coords=Val(true),pass_features=Val(false),pass_new_data=Val(true),kwargs...)
# Combined: κ(x, y, a(x), a(y), v(x), v(y))
FullKernelFormatter(;kwargs...) = DataFormatter(;pass_coords=Val(true),pass_features=Val(true),pass_new_data=Val(true),kwargs...)

# input
function fetch_data(::DataFormatter{false,false},(coords,features)::NamedTuple)
  @notimplemented "Wait ... no input?"
end

# input
function fetch_data(d::DataFormatter{true,false},(coords,features)::NamedTuple)
  (
    coords = sample(d.sampler,coords),
    features = nothing
  )
end

# input
function fetch_data(d::DataFormatter{false,true},(coords,features)::NamedTuple)
  (
    coords = nothing,
    features = sample(d.sampler,features)
  )
end

# input
function fetch_data(d::DataFormatter{true,true},(coords,features)::NamedTuple)
  (
    coords = sample(d.sampler,coords),
    features = sample(d.sampler,features)
  )
end

# output 
function fetch_data(d::DataFormatter,values::AbstractInputData)
  sample(d.sampler,values)
end

function fetch_stats(d::DataFormatter,data)
  Normalisation(data;normalise=d.normalise)
end

function merge_data(d::DataFormatter,(coords,features)::NamedTuple)
  _merge_data(coords,features)
end

function merge_data(::DataFormatter{A,B,C},fixed_data,new_data,coord_dims::Int=0) where {A,B,C}
  base = if coord_dims > 0 && size(fixed_data,1) > coord_dims
    if A && B
      fixed_data
    elseif A && !B
      view(fixed_data,1:coord_dims,:,:)
    elseif !A && B
      view(fixed_data,(coord_dims+1):size(fixed_data,1),:,:)
    else
      nothing
    end
  else
    (A || B) ? fixed_data : nothing
  end
  C ? _merge_data(base,new_data) : base
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
  features,comp_offsets = _init_data(a,x)
  @inbounds for (j,aj) in enumerate(a.bundle)
    af = lazy_map(aj,x)
    _set_data!(features,af,comp_offsets[j])
  end
  return features
end

function _get_values(ah::ParamFEFunction,x::AbstractVector{<:Point})
  values,comp_offsets = _init_data(ah,x)
  a = lazy_map(ah,x)
  _set_data!(values,a,first(comp_offsets))
  values
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
  for (i,f) in enumerate(a.bundle)
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
_merge_data(a::AbstractInputData,b::AbstractInputData) = cat(a,b;dims=1)
