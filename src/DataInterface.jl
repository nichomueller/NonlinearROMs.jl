struct Sampler{A<:Union{AbstractVector,Colon}}
  strategy::A
end

Sampler() = Sampler(:)

sample(s::Sampler,data::AbstractInputData) = view(data,:,s.strategy,:)

struct FeatureBundle{N,T<:NTuple{N,AbstractParamFunction}}
  bundle::T
end

function ParamDataStructures.num_params(a::FeatureBundle)
  dict = OrderedDict{UInt,AbstractRealisation}()
  for f in a.bundle
    μ = get_params(f)
    dict[hash(μ)] = μ
  end
  np = 0
  for μ in values(dict)
    np += num_params(μ)
  end
  return np
end

ParamDataStructures.num_params(a::ParamFEFunction) = param_length(a)
ParamDataStructures.num_params(a::TransientParamFESolution) = num_params(a.odesol.r)

struct DataFormatter{A,B,C,D,E<:NormStyle}
  pass_coords::Val{A}
  pass_features::Val{B}
  pass_new_data::Val{C}
  norm_style::E
  sampler::Sampler{D}
end

function DataFormatter(;
  pass_coords=Val{true}(),
  pass_features=Val{true}(),
  pass_new_data=Val{false}(),
  norm_style=NormStyle(),
  sampler=Sampler()
  )

  DataFormatter(pass_coords,pass_features,pass_new_data,norm_style,sampler)
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
  Normalisation(d.norm_style,data)
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

# concatenate along the channel axis (the first one)
function cat_channels(ldata,args...)
  offsets = compute_offsets(ldata;axis=1)
  cat_channels(ldata,offsets,args...)
end

function cat_channels(
  ldata::Tuple{Vararg{AbstractInputData{T}}},
  offsets::AbstractVector{<:Integer},
  lcoord_map::Table=_trivial_table(ldata),
  N::Int=maximum(lcoord_map.data)
  ) where T

  K = size(first(ldata),3)
  @check all(size(a,3) == K for a in ldata) "All local data must share the same samples"
  @check length(ldata) == length(lcoord_map) "Local data and local coordinate maps must have the same length"
  D = sum(a -> size(a,1),ldata)
  gdata = zeros(T,D,N,K)
  cache = array_cache(lcoord_map)
  @inbounds for l in eachindex(ldata)
    a = ldata[l]
    coord_ids = getindex!(cache,lcoord_map,l)
    for (p,ap) in pairs(a)
      i,j,k = Tuple(p)
      gdata[offsets[l]-1+i,coord_ids[j],k] = ap
    end
  end
  gdata
end

function get_channel(
  gdata::AbstractInputData{T},
  range::AbstractVector{<:Integer},
  gcoord_map::AbstractVector{<:Integer}
  ) where T

  d = length(range)
  n = length(gcoord_map)
  K = size(gdata,3)
  ldata = zeros(T,d,n,K)
  @inbounds for p in CartesianIndices(ldata)
    i,j,k = Tuple(p)
    ldata[i,j,k] = gdata[range[i],gcoord_map[j],k]
  end
  ldata
end

# concatenate along the sample axis (the last one)
function cat_samples(ldata,args...)
  offsets = compute_offsets(ldata;axis=3)
  cat_samples(ldata,offsets,args...)
end

function cat_samples(
  ldata::Tuple{Vararg{AbstractInputData{T}}},
  lcoord_map::Table=_trivial_table(ldata),
  N::Int=maximum(lcoord_map.data)
  ) where T

  D = size(first(ldata),1)
  @check all(size(a,1) == D for a in ldata) "All local data must have the same number of channels"
  offsets = ones(Int32,length(ldata)+1)
  for (l,a) in enumerate(ldata)
    offsets[l+1] = offsets[l] + size(a,3)
  end
  gdata = zeros(T,D,N,last(offsets)-1)
  cache = array_cache(lcoord_map)
  @inbounds for l in eachindex(ldata)
    a = ldata[l]
    coord_ids = getindex!(cache,lcoord_map,l)
    range = _get_range(offsets,l)
    for (p,ap) in pairs(a)
      i,j,k = Tuple(p)
      gdata[i,coord_ids[j],range[k]] = ap
    end
  end
  gdata,offsets
end

function get_sample(
  gdata::AbstractInputData{T},
  range::AbstractVector{<:Integer},
  gcoord_map::AbstractVector{<:Integer}
  ) where T

  D = size(gdata,1)
  n = length(gcoord_map)
  k = length(range)
  ldata = zeros(T,D,n,k)
  @inbounds for p in CartesianIndices(ldata)
    i,j,k = Tuple(p)
    ldata[i,j,k] = gdata[i,gcoord_map[j],range[k]]
  end
  ldata
end

function compute_offsets(a;axis=1)
  offsets = ones(Int32,length(a)+1)
  for (l,al) in enumerate(a)
    offsets[l+1] = offsets[l] + size(al,axis)
  end
  return offsets
end

# utils

function points_to_coords(x::AbstractVector{<:Point},K::Int=1)
  D = num_components(eltype(x))
  N = length(x)
  coords = zeros(TData,D,N,K)
  @inbounds @views for (i,xi) in enumerate(x)
    for d in 1:D 
      coords[d,i,:] .= xi.data[d]
    end
  end
  return coords
end

function coords_to_points(coords::AbstractInputData)
  D = size(coords,1)
  points = Vector{Point{D,TData}}(undef,size(coords,2))
  @inbounds for i in axes(coords,2)
    points[i] = Point(ntuple(d -> coords[d,i,1],Val(D)))
  end
  points
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
  D = offsets[end]-1
  N = length(x)
  K = num_params(a)
  data = zeros(TData,D,N,K)
  return data,offsets
end

function _offsets(a::FeatureBundle,x::Point)
  offsets = zeros(Int,length(a.bundle)+1)
  for (i,f) in enumerate(a.bundle)
    T = return_type(f,x)
    offsets[i+1] = num_components(eltype(T))
  end
  length_to_ptrs!(offsets)
  return offsets
end

function _offsets(a::ParamFEFunction,x::Point)
  offsets = zeros(Int,2)
  T = return_type(a,x)
  offsets[2] = num_components(eltype(T))
  length_to_ptrs!(offsets)
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

_set_data!(data,valsik,o,i,k) = (data[o,i,k] = valsik)

function _set_data!(data,valsik::Point{D},o,i,k) where D
  for d in 1:D
    data[o-1+d,i,k] = valsik.data[d]
  end
end

_merge_data(a,b) = @abstractmethod
_merge_data(a::Nothing,b::Nothing) = nothing
_merge_data(a::Nothing,b) = b
_merge_data(a,b::Nothing) = a
_merge_data(a::AbstractInputData,b::AbstractInputData) = cat(a,b;dims=1)

function _trivial_table(a::Tuple{Vararg{AbstractInputData{T}}}) where T
  ptrs = zeros(Int,length(a)+1)
  @inbounds for (i,ai) in enumerate(a)
    ptrs[i+1] = ptrs[i] + size(ai,2)
  end
  data = zeros(T,ptrs[end])
  length_to_ptrs!(ptrs)
  @inbounds for i in eachindex(a)
    pini = ptrs[i]
    pend = ptrs[i+1]-1
    for (k,p) in enumerate(pini:pend)
      data[p] = k
    end
  end
  Table(data,ptrs)
end

_get_range(::Nothing,::Int) = Colon()
_get_range(l_to_sample_offset::AbstractVector,l::Int) = l_to_sample_offset[l]:(l_to_sample_offset[l+1]-1)