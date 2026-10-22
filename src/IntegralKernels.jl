abstract type AbstractIntegralKernel <: AbstractLuxLayer end

get_in_dimension(k::AbstractLuxLayer) = @abstractmethod
get_out_dimension(k::AbstractLuxLayer) = @abstractmethod

get_in_dimension(k::Lux.Dense) = k.in
get_out_dimension(k::Lux.Dense) = k.out
get_in_dimension(k::Lux.Conv) = k.in
get_out_dimension(k::Lux.Conv) = k.out

# we potentially want to replace the space with a new one during the online phase
# thus we make this struct mutable
mutable struct IntegralKernel{A<:AbstractLuxLayer,B<:InducedMeasure,C<:InputData,D<:DataFormatter} <: AbstractIntegralKernel
  const kernel::A
  measure::B
  formatter::C
end

get_in_dimension(k::IntegralKernel) = get_in_dimension(k.kernel)
get_out_dimension(k::IntegralKernel) = get_out_dimension(k.kernel)

function Lux.initialparameters(rng::AbstractRNG,l::IntegralKernel)
  Lux.initialparameters(rng,l.kernel)
end

function Lux.initialstates(rng::AbstractRNG,l::IntegralKernel)
  Lux.initialstates(rng,l.kernel)
end

function (l::IntegralKernel)(x,ps,st)
  v = fetch_data(l.formatter,x)
  vals,st = l.kernel(v,ps,st)
  y = integration(k_v(vals,x),l.measure)
  return y,st
end

struct FourierKernel <: AbstractIntegralKernel
  in_dims::Int
  out_dims::Int
  truncation::Int
end

get_in_dimension(k::FourierKernel) = k.in_dims
get_out_dimension(k::FourierKernel) = k.out_dims

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
  activation::B
end

function KernelNeuralLayer(kernel::AbstractIntegralKernel;activation=identity,kwargs...)
  KernelNeuralLayer(kernel,activation)
end

function Lux.initialparameters(rng::AbstractRNG,l::KernelNeuralLayer)
  in_dims = get_in_dimension(l.kernel)
  out_dims = get_out_dimension(l.kernel)
  return (
    kernel = Lux.initialparameters(rng,l.kernel),
    weights = Lux.glorot_uniform(rng,out_dims,in_dims),
    bias = zeros(TData,out_dims),
  )
end

function Lux.initialstates(rng::AbstractRNG,l::KernelNeuralLayer)
  return (
    kernel = Lux.initialstates(rng,l.kernel),
  )
end

function (l::KernelNeuralLayer)(x,ps,st)
  xk = merge_data(l.formatter,(l.fixed_data,x))
  kout,kst = l.kernel(xk,ps.kernel,st.kernel)
  wout = ps.weights*x
  y = l.activation.(wout .+ kout .+ ps.bias)
  return y,(kernel=kst,)
end

# utils


function k_v(k::AbstractArray{T,4},v::AbstractArray{T,3}) where T
  K = reshape(permutedims(k,(1,4,2,3)),size(k,1),:)
  V = vec(v)
  return K * V
end

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