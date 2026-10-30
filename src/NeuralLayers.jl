
"""
    struct Lift{A<:AbstractLuxLayer} <: AbstractLuxLayer

Adapts a plain local layer (e.g. `Lux.Dense`) to a [`KernelNeuralModel`](@ref)'s
chain: called as `l(fixed_data,ps,st)`, it runs `layer` on `fixed_data` alone (no
running state exists yet) and injects the result into the `(fixed_data,v)` tuple
convention every [`KernelNeuralLayer`](@ref) downstream expects.
"""
struct Lift{A<:AbstractLuxLayer} <: AbstractLuxLayer
  layer::A
end

Lux.initialparameters(rng::AbstractRNG,l::Lift) = Lux.initialparameters(rng,l.layer)
Lux.initialstates(rng::AbstractRNG,l::Lift) = Lux.initialstates(rng,l.layer)

function (l::Lift)(fixed_data,ps,st)
  v,st = l.layer(fixed_data,ps,st)
  return (fixed_data,v),st
end

"""
    struct Project{A<:AbstractLuxLayer} <: AbstractLuxLayer

The mirror of [`Lift`](@ref): called as `l((fixed_data,v),ps,st)`, it drops
`fixed_data` and runs `layer` on `v` alone, turning the final hidden state into
a plain prediction matching the training target's shape.
"""
struct Project{A<:AbstractLuxLayer} <: AbstractLuxLayer
  layer::A
end

Lux.initialparameters(rng::AbstractRNG,l::Project) = Lux.initialparameters(rng,l.layer)
Lux.initialstates(rng::AbstractRNG,l::Project) = Lux.initialstates(rng,l.layer)

function (l::Project)((fixed_data,v),ps,st)
  l.layer(v,ps,st)
end

"""
    RoPELayer(in_dims, out_dims; kwargs...)

Dense projection followed by rotary positional encoding. The input is a
`(features, coordinates)` tuple, where features have shape
`(in_dims × npoints)` and coordinates have shape `(ndims × npoints)` (a vector
is accepted for one-dimensional coordinates). Each adjacent feature pair is
rotated by a phase derived from the corresponding point's physical coordinates;
an odd final feature is left unchanged.
"""
struct RoPELayer{A<:AbstractLuxLayer,B<:AbstractVector} <: AbstractLuxLayer
  layer::A
  inv_freq::B
end

function RoPELayer(in_dims::Int,out_dims::Int;kwargs...)
  layer = Dense(in_dims=>out_dims;kwargs...)
  inv_freq = zeros(TData,div(out_dims,2))
  for i in eachindex(inv_freq)
    inv_freq[i] = 10000^(-2*(i-1)/out_dims)
  end
  return RoPELayer(layer,inv_freq)
end

function Lux.initialparameters(rng::AbstractRNG,l::RoPELayer)
  return (
    layer = Lux.initialparameters(rng,l.layer),
  )
end

function Lux.initialstates(rng::AbstractRNG,l::RoPELayer)
  return (
    layer = Lux.initialstates(rng,l.layer),
  )
end

function (l::RoPELayer)(x::Tuple,ps,st)
  @check length(x) == 2 "RoPELayer input must be a (features, coordinates) tuple"
  features,coordinates = x
  y,st_layer = l.layer(features,ps.layer,st.layer)
  z = _apply_rope(y,coordinates,l.inv_freq)
  return z,(layer=st_layer,)
end

# utils 

function _apply_rope(y::AbstractMatrix,coordinates::AbstractVector,inv_freq)
  _apply_rope(y,reshape(coordinates,1,:),inv_freq)
end

function _apply_rope(y::AbstractMatrix,coordinates::AbstractMatrix,inv_freq)
  npoints = size(y,2)
  @check size(coordinates,2) == npoints "Coordinates must have one column per point"
  @check size(coordinates,1) > 0 "At least one coordinate dimension is required"
  @check div(size(y,1),2) == length(inv_freq) "There must be one rotary frequency per feature pair"
  z = similar(y)
  @inbounds for k in eachindex(inv_freq)
    i = 2*k-1
    axis = mod1(k,size(coordinates,1))
    θ = coordinates[axis,:] .* inv_freq[k]
    c = cos.(θ)
    s = sin.(θ)
    @views z[i,:] .= c .* y[i,:] .- s .* y[i+1,:]
    @views z[i+1,:] .= s .* y[i,:] .+ c .* y[i+1,:]
  end
  if isodd(size(y,1))
    @views z[end,:] .= y[end,:]
  end
  return z
end