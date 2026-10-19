"""
    RoPELayer(in_dims, out_dims; kwargs...)

Dense projection followed by rotary positional encoding. The input is a
`(features, coordinates)` tuple, where features have shape
`(in_dims × npoints)` and coordinates have shape `(ndims × npoints)` (a vector
is accepted for one-dimensional coordinates). Each adjacent feature pair is
rotated by a phase derived from the corresponding point's physical coordinates;
an odd final feature is left unchanged.
"""
struct RoPELayer{A<:AbstractLuxLayer,T<:AbstractVector} <: AbstractLuxLayer
  layer::A
  inverse_frequencies::T
end

function RoPELayer(in_dims::Int,out_dims::Int;kwargs...)
  layer = Dense(in_dims=>out_dims;kwargs...)
  inverse_frequencies = Float32[10000^(-2*(k-1)/out_dims) for k in 1:div(out_dims,2)]
  return RoPELayer(layer,inverse_frequencies)
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
  z = _apply_rope(y,coordinates,l.inverse_frequencies)
  return z,(layer=st_layer,)
end

function _apply_rope(y::AbstractMatrix,coordinates::AbstractVector,inverse_frequencies)
  _apply_rope(y,reshape(coordinates,1,:),inverse_frequencies)
end

function _apply_rope(y::AbstractMatrix,coordinates::AbstractMatrix,inverse_frequencies)
  npoints = size(y,2)
  @check size(coordinates,2) == npoints "Coordinates must have one column per point"
  @check size(coordinates,1) > 0 "At least one coordinate dimension is required"
  @check div(size(y,1),2) == length(inverse_frequencies) "There must be one rotary frequency per feature pair"
  z = similar(y)
  @inbounds for k in eachindex(inverse_frequencies)
    i = 2*k-1
    axis = mod1(k,size(coordinates,1))
    θ = coordinates[axis,:] .* inverse_frequencies[k]
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

abstract type AbstractIntegralKernel <: AbstractLuxLayer end

# we potentially want to replace the space with a new one during the online phase
# thus we make this struct mutable
mutable struct IntegralKernel{A<:AbstractLuxLayer,B<:InducedMeasure,C<:LayerData} <: AbstractIntegralKernel
  const kernel::A
  measure::B
  formatter::C
end

function Lux.initialparameters(rng::AbstractRNG,l::IntegralKernel)
  Lux.initialparameters(rng,l.kernel)
end

function Lux.initialstates(rng::AbstractRNG,l::IntegralKernel)
  Lux.initialstates(rng,l.kernel)
end

function (l::IntegralKernel)(x,ps,st)
  v = gather_data(l.formatter,x)
  vals,st = l.kernel(v,ps,st)
  y = integration(vals,l.measure)
  return y,st
end

struct FourierKernel <: AbstractIntegralKernel
  in_dims::Int
  out_dims::Int
  truncation::Int
end

function Lux.initialparameters(rng::AbstractRNG,l::FourierKernel)
  return (
    kernel = rand(rng,Uniform(-1,1),l.truncation,l.out_dims,l.in_dims),
  )
end

function Lux.initialstates(rng::AbstractRNG,l::FourierKernel)
  return (
    kernel = NamedTuple(),
  )
end

function (l::FourierKernel)(x,ps,st)
  x̂ = rfft(x)
  ŷ = fapply(ps.kernel,x̂)
  y = irfft(ŷ)
  return y,st
end

mutable struct AttentionKernel{A<:AbstractLuxLayer,B<:AbstractLuxLayer,C<:InducedMeasure} <: AbstractIntegralKernel
  const score::A
  const value::B
  measure::C
end

function Lux.initialparameters(rng::AbstractRNG,l::AttentionKernel)
  return (
    score = Lux.initialparameters(rng,l.score),
    value = Lux.initialparameters(rng,l.value),
  )
end

function Lux.initialstates(rng::AbstractRNG,l::AttentionKernel)
  return (
    score = Lux.initialstates(rng,l.score),
    value = Lux.initialstates(rng,l.value),
  )
end

function (l::AttentionKernel)(x,ps,st)
  s,st_score = l.score(x,ps.score,st.score)
  val,st_value = l.value(x,ps.value,st.value)
  y = attention_integral(s,val,l.measure)
  return y,(score=st_score,value=st_value)
end

struct MultipleAttentionKernel{N,A<:NTuple{N,AbstractLuxLayer},B<:NTuple{N,AbstractLuxLayer},C<:AbstractLuxLayer,D<:InducedMeasure} <: AbstractIntegralKernel
  const scores::A
  const values::B
  const aggregation::C
  measure::D
end

function Lux.initialparameters(rng::AbstractRNG,l::MultipleAttentionKernel{N}) where N
  return (
    score = ntuple(i -> Lux.initialparameters(rng,l.scores[i]),Val(N)),
    value = ntuple(i -> Lux.initialparameters(rng,l.values[i]),Val(N)),
    aggregation = Lux.initialparameters(rng,l.aggregation),
  )
end

function Lux.initialstates(rng::AbstractRNG,l::MultipleAttentionKernel{N}) where N
  return (
    score = ntuple(i -> Lux.initialstates(rng,l.scores[i]),Val(N)),
    value = ntuple(i -> Lux.initialstates(rng,l.values[i]),Val(N)),
    aggregation = Lux.initialstates(rng,l.aggregation),
  )
end

function (l::MultipleAttentionKernel{N})(x,ps,st) where N
  st_score = ()
  st_value = ()
  y = ()
  for i in 1:N
    si,st_scorei = l.scores[i](x,ps.score[i],st.score[i])
    vali,st_valuei = l.values[i](x,ps.value[i],st.value[i])
    st_score = (st_score...,st_scorei)
    st_value = (st_value...,st_valuei)
    y = (y...,attention_integral(si,vali,l.measure))
  end
  z,st_aggregation = l.aggregation(hcat(y...),ps.aggregation,st.aggregation)
  return z,(score=st_score,value=st_value,aggregation=st_aggregation)
end

# implement the attention layer + RoPE

"""
    struct KernelNeuralLayer{A<:AbstractIntegralKernel,B} <: AbstractLuxLayer
      kernel::A
      in_dims::Int
      out_dims::Int
      activation::B
    end

One iterative layer of a kernel neural operator (Kovachki et al.):
```
v_{t+1}(x) = σ(W v_t(x) + (K_t v_t)(x) + b(x))
```
`kernel` (`K_t`) is the non-local integral operator -- any `AbstractIntegralKernel`
(`IntegralKernel` wrapping a graph/FE-quadrature `InducedMeasure`, `FourierKernel`,
`AttentionKernel`, ...). The pointwise local linear term `W`, bias `b`, and
nonlinearity `σ` are owned by this layer, not by `kernel`, since they're the same
shape/role regardless of which non-local discretization `kernel` uses.
"""
struct KernelNeuralLayer{A<:AbstractIntegralKernel,B<:Function} <: AbstractLuxLayer
  kernel::A
  in_dims::Int
  out_dims::Int
  activation::B
end

function KernelNeuralLayer(
  kernel::AbstractIntegralKernel,
  in_dims::Int,
  out_dims::Int;
  activation=identity,
  kwargs...
  )

  KernelNeuralLayer(kernel,in_dims,out_dims,activation)
end

function KernelNeuralLayer(kernel::AbstractIntegralKernel,dims::Pair,args...;kwargs...)
  in_dims,out_dims = dims
  KernelNeuralLayer(kernel,in_dims,out_dims,args...;kwargs...)
end

function Lux.initialparameters(rng::AbstractRNG,l::KernelNeuralLayer)
  return (
    kernel = Lux.initialparameters(rng,l.kernel),
    weights = Lux.glorot_uniform(rng,l.out_dims,l.in_dims),
    bias = zeros(Float32,l.out_dims),
  )
end

function Lux.initialstates(rng::AbstractRNG,l::KernelNeuralLayer)
  return (
    kernel = Lux.initialstates(rng,l.kernel),
  )
end

function (l::KernelNeuralLayer)(x,ps,st)
  kout,kst = l.kernel(x,ps.kernel,st.kernel)
  wout = ps.weights*x
  y = l.activation.(wout .+ kout .+ ps.bias)
  return y,(kernel=kst,)
end

# utils

function fapply(W::AbstractArray{T,3},x::AbstractMatrix) where T
  k,in_dims,out_dims = size(W)
  A = reshape(permutedims(W,(2,1,3)),k*out_dims,in_dims)
  b = vec(x)
  reshape(A*b,k,out_dims)
end

function attention_integral(scores,values,meas::InducedMeasure)
  @notimplemented "Attention integration is not implemented for measure type $(typeof(meas))."
end

function attention_integral(scores,values,meas::GraphMeasure)
  graph = get_space(meas)
  α = attention(vec(scores),graph)
  scatter_data(α'.*values,graph)
end