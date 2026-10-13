abstract type KernelLayer <: Lux.AbstractLuxLayer end

function Lux.initialstates(rng::Random.AbstractRNG,l::KernelLayer)
  return (
    kernel = NamedTuple(),
  )
end

struct NeuralKernel{A<:NeuralModel} <: KernelLayer
  model::A
end

function Lux.initialparameters(rng::Random.AbstractRNG,l::NeuralKernel)
  return Lux.initialparameters(rng,l.model)
end

function (l::NeuralKernel)(x,ps,st)
  l.model(x,ps,st)
end

abstract type AbstractIntegralKernel <: Lux.AbstractLuxLayer end

# we potentially want to replace the space with a new one during the online phase
# thus we make this struct mutable
mutable struct IntegralKernel{A<:KernelLayer,B<:InducedMeasure} <: AbstractIntegralKernel
  const kernel::A
  measure::B
end

function Lux.initialparameters(rng::Random.AbstractRNG,l::IntegralKernel)
  Lux.initialparameters(rng,l.kernel)
end

function Lux.initialstates(rng::Random.AbstractRNG,l::IntegralKernel)
  Lux.initialstates(rng,l.kernel)
end

function (l::IntegralKernel)(v,ps,st)
  # `v` is the raw hidden state (channels x n_points); prepare_kernel_input turns it
  # into whatever num_coords(l.measure)-indexed input l.kernel actually needs (the
  # identity for FE-quadrature, the per-edge [v(sender);v(receiver)] pair for a
  # graph). vals is dout x Nh (channels x batch, the Lux/NeuralModel convention):
  # Nh is the last axis, and it's what gets integrated over (next line).
  x = prepare_kernel_input(v,l.measure)
  vals,st = l.kernel(x,ps,st)
  y = integrate_over(vals,l.measure)
  return y,st
end

struct FourierKernel <: AbstractIntegralKernel
  in_channels::Int
  out_channels::Int
  truncation::Int
end

function Lux.initialparameters(rng::Random.AbstractRNG,l::FourierKernel)
  return (
    kernel = rand(rng,Uniform(-1,1),l.truncation,l.out_channels,l.in_channels),
  )
end

function (l::FourierKernel)(x,ps,st)
  x̂ = rfft(x)
  ŷ = fapply(ps.kernel,x̂)
  y = irfft(ŷ)
  return y,st
end

"""
    struct AttentionKernel{A<:KernelLayer,B<:KernelLayer,C<:AbstractGraph} <: AbstractIntegralKernel
      score::A
      value::B
      graph::C
    end

Data-dependent (learned) aggregation kernel: Kovachki et al.'s nonlinear case
`κ(x,y,v(x),v(y))`, but with the measure itself learned rather than fixed -- the
building block for attention/transformer-style neural operators (HAMLET = GNO +
attention). `x` (typically `gather_edge_features(v,graph)`) is evaluated by both
sub-kernels: `score` produces one scalar compatibility score per edge, `value`
produces the per-edge message to be aggregated. Every vertex's incoming edges are
softmax-normalized (`softmax_weights`) before the weighted sum, replacing the
fixed `1/degree` averaging a plain `IntegralKernel`+graph `InducedMeasure` uses.
"""
struct AttentionKernel{A<:KernelLayer,B<:KernelLayer,C<:AbstractGraph} <: AbstractIntegralKernel
  score::A
  value::B
  graph::C
end

function Lux.initialparameters(rng::Random.AbstractRNG,l::AttentionKernel)
  return (
    score = Lux.initialparameters(rng,l.score),
    value = Lux.initialparameters(rng,l.value),
  )
end

function Lux.initialstates(rng::Random.AbstractRNG,l::AttentionKernel)
  return (
    score = Lux.initialstates(rng,l.score),
    value = Lux.initialstates(rng,l.value),
  )
end

function (l::AttentionKernel)(v,ps,st)
  x = prepare_kernel_input(v,l.graph)
  s,st_score = l.score(x,ps.score,st.score)
  val,st_value = l.value(x,ps.value,st.value)
  α = softmax_weights(vec(s),l.graph)
  y = scatter_by_dst(α' .* val,l.graph)
  return y,(score=st_score,value=st_value)
end

"""
    struct KernelNeuralLayer{A<:AbstractIntegralKernel,B} <: Lux.AbstractLuxLayer
      kernel::A
      din::Int
      dout::Int
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

Only genuinely composes as a per-`x` layer today for graph-based kernels (one
output per vertex, via `_integrate_over(a,::AbstractGraph)`) and `FourierKernel`.
`IntegralKernel` wrapping an FE-quadrature `InducedMeasure` collapses to a single
scalar per channel (`_integrate_over(a,::SimpleFESpace)` integrates the whole
domain down to one number) rather than one value per query point, so it only
makes sense here for a kernel with no real `x`-dependence or as a global
pooling/projection term -- not as a general per-`x` kernel layer. Giving FE
quadrature a proper per-point output (dense pairwise quadrature, not just a
single aggregate) hasn't been done yet.
"""
struct KernelNeuralLayer{A<:AbstractIntegralKernel,B} <: Lux.AbstractLuxLayer
  kernel::A
  din::Int
  dout::Int
  activation::B
end

function KernelNeuralLayer(kernel::AbstractIntegralKernel,din::Int,dout::Int;activation=identity)
  KernelNeuralLayer(kernel,din,dout,activation)
end

function Lux.initialparameters(rng::Random.AbstractRNG,l::KernelNeuralLayer)
  return (
    kernel = Lux.initialparameters(rng,l.kernel),
    weights = Lux.glorot_uniform(rng,l.dout,l.din),
    bias = zeros(Float32,l.dout),
  )
end

function Lux.initialstates(rng::Random.AbstractRNG,l::KernelNeuralLayer)
  return (
    kernel = Lux.initialstates(rng,l.kernel),
  )
end

function (l::KernelNeuralLayer)(v,ps,st)
  kout,kst = l.kernel(v,ps.kernel,st.kernel)
  wout = ps.weights*v
  y = l.activation.(wout .+ kout .+ ps.bias)
  return y,(kernel=kst,)
end

# utils

function fapply(W::AbstractArray{T,3},x::AbstractMatrix) where T
  k,din,dout = size(W)
  A = reshape(permutedims(W,(2,1,3)),k*dout,din)
  b = vec(x)
  reshape(A*b,k,dout)
end