struct RoPELayer{A<:AbstractLuxLayer,T} <: AbstractLuxLayer
  layer::A
  rotation::Matrix{T}
end

function RoPELayer(in_dims::Int,out_dims::Int;kwargs...)
  layer = Dense(in_dims=>out_dims;kwargs...)
  rotation = _rotations_matrix(Float32,out_dims)
  return RoPELayer(layer,rotation)
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

function (l::RoPELayer)(x,ps,st)
  y,st_layer = l.layer(x,ps.layer,st.layer)
  z = rotation * y
  return z,(layer=st_layer,)
end

abstract type AbstractIntegralKernel <: AbstractLuxLayer end

get_measure(l::AbstractIntegralKernel) = @abstractmethod
get_space(l::AbstractIntegralKernel) = get_space(get_measure(l))

# we potentially want to replace the space with a new one during the online phase
# thus we make this struct mutable
mutable struct IntegralKernel{A<:AbstractLuxLayer,B<:InducedMeasure} <: AbstractIntegralKernel
  const kernel::A
  measure::B
end

function Lux.initialparameters(rng::AbstractRNG,l::IntegralKernel)
  Lux.initialparameters(rng,l.kernel)
end

function Lux.initialstates(rng::AbstractRNG,l::IntegralKernel)
  Lux.initialstates(rng,l.kernel)
end

function (l::IntegralKernel)(x,ps,st)
  vals,st = l.kernel(x,ps,st)
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
struct KernelNeuralLayer{A<:AbstractIntegralKernel,B<:Function,C<:DataFormatter} <: AbstractLuxLayer
  kernel::A
  in_dims::Int
  out_dims::Int
  activation::B
  formatter::C
end

function KernelNeuralLayer(
  kernel::AbstractIntegralKernel,
  in_dims::Int,
  out_dims::Int;
  activation=identity,
  kwargs...
  )

  formatter = DataFormatter(;kwargs...)
  KernelNeuralLayer(kernel,in_dims,out_dims,activation,formatter)
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
  v = get_layer_data(l,x)
  kout,kst = l.kernel(v,ps.kernel,st.kernel)
  wout = ps.weights*v
  y = l.activation.(wout .+ kout .+ ps.bias)
  return y,(kernel=kst,)
end

# utils

function _fill_block!(b,θ)
  @check size(b) == (2,2) "Rotation block must be 2x2"
  b[1,1] = cos(θ)
  b[1,2] = -sin(θ)
  b[2,1] = sin(θ)
  b[2,2] = cos(θ)
  b
end

function _fill_rotations!(mat,Θ)
  @inbounds @views for k in eachindex(Θ)
    b = mat[(k-1)*2+1:k*2,(k-1)*2+1:k*2]
    _fill_block!(b,Θ[k])
  end
  mat
end

function _rotations_matrix(::Type{T},d) where T
  Θ = zeros(T,div(d,2))
  for k in eachindex(Θ)
    Θ[k] = T(10000)^(-2*(k-1)/d)
  end
  mat = zeros(T,d,d)
  _fill_rotations!(mat,Θ)
end

function fapply(W::AbstractArray{T,3},x::AbstractMatrix) where T
  k,in_dims,out_dims = size(W)
  A = reshape(permutedims(W,(2,1,3)),k*out_dims,in_dims)
  b = vec(x)
  reshape(A*b,k,out_dims)
end

function attention_integral(scores,values,meas::InducedMeasure)
  @notimplemented "Provided a measure of type $(typeof(meas)); consider passing a `GraphInducedMeasure` instead."
end

function attention_integral(scores,values,meas::GraphInducedMeasure)
  α = attention(scores,meas.graph)
  scatter(α'.*values,meas.graph)
end

function get_layer_data(l::KernelNeuralLayer,x)
  gather_data(l.formatter,get_space(l.kernel),x)
end