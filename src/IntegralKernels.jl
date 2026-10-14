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

"""
    struct AttentionKernel{A<:AbstractLuxLayer,B<:AbstractLuxLayer,C<:AbstractGraph} <: AbstractIntegralKernel
      score::A
      value::B
      graph::C
    end

Data-dependent (learned) aggregation kernel: Kovachki et al.'s nonlinear case
`κ(x,y,v(x),v(y))`, but with the measure itself learned rather than fixed -- the
builin_dimsg block for attention/transformer-style neural operators (HAMLET = GNO +
attention). `x` (typically `gather_edge_features(v,graph)`) is evaluated by both
sub-kernels: `score` produces one scalar compatibility score per edge, `value`
produces the per-edge message to be aggregated. Every vertex's incoming edges are
softmax-normalized (`attention`) before the weighted sum, replacing the
fixed `1/degree` averaging a plain `IntegralKernel`+graph `InducedMeasure` uses.
"""
struct AttentionKernel{A<:AbstractLuxLayer,B<:AbstractLuxLayer,C<:AbstractGraph} <: AbstractIntegralKernel
  score::A
  value::B
  graph::C
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
  α = attention(vec(s),l.graph)
  y = scatter(α' .* val,l.graph)
  return y,(score=st_score,value=st_value)
end

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

function fapply(W::AbstractArray{T,3},x::AbstractMatrix) where T
  k,in_dims,out_dims = size(W)
  A = reshape(permutedims(W,(2,1,3)),k*out_dims,in_dims)
  b = vec(x)
  reshape(A*b,k,out_dims)
end

function get_layer_data(l::KernelNeuralLayer,x)
  gather_data(l.formatter,get_space(l.kernel),x)
end