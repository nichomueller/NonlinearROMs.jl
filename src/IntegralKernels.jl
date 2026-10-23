abstract type AbstractIntegralKernel <: AbstractLuxLayer end

get_in_dimension(k::AbstractLuxLayer) = @abstractmethod
get_out_dimension(k::AbstractLuxLayer) = @abstractmethod

get_in_dimension(k::Lux.Dense) = k.in
get_out_dimension(k::Lux.Dense) = k.out
get_in_dimension(k::Lux.Conv) = k.in
get_out_dimension(k::Lux.Conv) = k.out

kernel_input(x,meas::InducedMeasure) = kernel_input(x,get_space(meas))
kernel_input(x,space::DiscreteSpace) = x
kernel_input(x,space::AbstractGraphInducedSpace) = gather_data(x,get_graph(space))

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

function (l::FourierKernel)((fixed_data,x),ps,st)
  x̂ = rfft(x)
  ŷ = fapply(ps.kernel,x̂)
  y = irfft(ŷ)
  return (fixed_data,y),st
end

# we potentially want to replace the space with a new one during the online phase
# thus we make this struct mutable
"""
    mutable struct IntegralKernel{A<:AbstractLuxLayer,B<:InducedMeasure,C<:DataFormatter} <: AbstractIntegralKernel
      const kernel::A
      measure::B
      formatter::C
    end

Wraps a plain network `kernel` into a non-local integral operator
`(K v)(x) = ∫ κ_θ(x,y,...) v(y) dy`, discretised by `measure`. Called as
`l((fixed_data,v),ps,st)`: `fixed_data` is the per-sample-fixed `(x,a(x))` tensor
assembled once upstream (see [`DataInterface.fetch_data`](@ref)), `v` is the
running hidden state from the previous layer. `formatter` (its own, layer-local
[`DataFormatter`](@ref)) decides which of `(x,a(x))`/`v` actually get concatenated
and fed to `kernel` -- different layers legitimately want different subsets
(Kovachki et al.'s three kernel cases `κ(x,y)`/`κ(x,y,a(x),a(y))`/`κ(x,y,v(x),v(y))`
are exactly a choice of `formatter`). `fixed_data` passes through unchanged so the
next layer can use it too.
"""
mutable struct IntegralKernel{A<:AbstractLuxLayer,B<:InducedMeasure,C<:DataFormatter} <: AbstractIntegralKernel
  const kernel::A
  measure::B
  const formatter::C
end

get_in_dimension(k::IntegralKernel) = get_in_dimension(k.kernel)
get_out_dimension(k::IntegralKernel) = get_out_dimension(k.kernel)

function Lux.initialparameters(rng::AbstractRNG,l::IntegralKernel)
  Lux.initialparameters(rng,l.kernel)
end

function Lux.initialstates(rng::AbstractRNG,l::IntegralKernel)
  Lux.initialstates(rng,l.kernel)
end

function (l::IntegralKernel)((fixed_data,v),ps,st)
  x = merge_data(l.formatter,fixed_data,v)
  xk = kernel_input(x,l.measure)
  vals,st = l.kernel(xk,ps,st)
  y = integration(vals,l.measure)
  return (fixed_data,y),st
end

mutable struct AttentionKernel{A<:AbstractLuxLayer,B<:AbstractLuxLayer,C<:InducedMeasure,D<:DataFormatter} <: AbstractIntegralKernel
  const score::A
  const value::B
  measure::C
  const formatter::D
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

function (l::AttentionKernel)((fixed_data,v),ps,st)
  x = merge_data(l.formatter,fixed_data,v)
  xk = kernel_input(x,l.measure)
  s,st_score = l.score(xk,ps.score,st.score)
  val,st_value = l.value(xk,ps.value,st.value)
  y = attention_integral(s,val,l.measure)
  return (fixed_data,y),(score=st_score,value=st_value)
end

struct MultipleAttentionKernel{N,A<:NTuple{N,AbstractLuxLayer},B<:NTuple{N,AbstractLuxLayer},C<:AbstractLuxLayer,D<:InducedMeasure,E<:DataFormatter} <: AbstractIntegralKernel
  const scores::A
  const values::B
  const aggregation::C
  measure::D
  const formatter::E
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

function (l::MultipleAttentionKernel{N})((fixed_data,v),ps,st) where N
  x = merge_data(l.formatter,fixed_data,v)
  xk = kernel_input(x,l.measure)
  st_score = ()
  st_value = ()
  y = ()
  for i in 1:N
    si,st_scorei = l.scores[i](xk,ps.score[i],st.score[i])
    vali,st_valuei = l.values[i](xk,ps.value[i],st.value[i])
    st_score = (st_score...,st_scorei)
    st_value = (st_value...,st_valuei)
    y = (y...,attention_integral(si,vali,l.measure))
  end
  z,st_aggregation = l.aggregation(hcat(y...),ps.aggregation,st.aggregation)
  return (fixed_data,z),(score=st_score,value=st_value,aggregation=st_aggregation)
end

# implement the attention layer + RoPE

"""
    struct KernelNeuralLayer{A<:AbstractIntegralKernel,B<:Function} <: AbstractLuxLayer
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
shape/role regardless of which non-local discretization `kernel` uses. Called as
`l((fixed_data,v),ps,st)`: `v` feeds both the local `W v` term and `kernel`
directly; `fixed_data` (the per-sample `(x,a(x))` tensor) is only relevant to
`kernel` (via its own `formatter`) and passes through untouched, so the next
layer in the stack can use it too.

`in_dims`/`out_dims` are the running hidden state's channel counts (`v_t`'s and
`v_{t+1}`'s), sizing `W`/`b` -- **not** `kernel`'s own internal input width.
`kernel`'s network sees `merge_data`'d/gathered data (often `v`'s channels plus
`fixed_data`'s, doubled again by `gather_data` for a graph measure), a completely
different, independently-sized tensor; `get_in_dimension(l.kernel)` reports that
internal width, not `v`'s, so it must not be used to size `W`.
"""
struct KernelNeuralLayer{A<:AbstractIntegralKernel,B<:Function} <: AbstractLuxLayer
  kernel::A
  in_dims::Int
  out_dims::Int
  activation::B
end

function KernelNeuralLayer(kernel::AbstractIntegralKernel,in_dims::Int,out_dims::Int;activation=identity)
  KernelNeuralLayer(kernel,in_dims,out_dims,activation)
end

function Lux.initialparameters(rng::AbstractRNG,l::KernelNeuralLayer)
  return (
    kernel = Lux.initialparameters(rng,l.kernel),
    weights = Lux.glorot_uniform(rng,l.out_dims,l.in_dims),
    bias = zeros(TData,l.out_dims),
  )
end

function Lux.initialstates(rng::AbstractRNG,l::KernelNeuralLayer)
  return (
    kernel = Lux.initialstates(rng,l.kernel),
  )
end

function (l::KernelNeuralLayer)((fixed_data,v),ps,st)
  (_,kout),kst = l.kernel((fixed_data,v),ps.kernel,st.kernel)
  wout = ps.weights*v
  y = l.activation.(wout .+ kout .+ ps.bias)
  return (fixed_data,y),(kernel=kst,)
end

# """
#     struct KernelOperator{N,A<:AbstractLuxLayer,B<:KernelNeuralLayer,C<:AbstractLuxLayer} <: AbstractLuxLayer
#       lifting::A
#       kernel_layers::NTuple{N,B}
#       projection::C
#     end

# The actual `Lux.AbstractLuxLayer` that runs a [`KernelNeuralModel`](@ref) recipe
# (built from one via `build_model`). Called as `l(fixed_data,ps,st)`, where
# `fixed_data` is the per-sample `(x,a(x))` tensor assembled once upstream (see
# [`DataInterface.fetch_data`](@ref)):
# 1. `lifting` maps `fixed_data` alone to the initial hidden state `v₀` (Kovachki
#    et al.'s `P`) -- there is no previous-layer state yet, so no tuple is needed here.
# 2. Each `kernel_layers[i]` is called as `(fixed_data,v)`, threading `fixed_data`
#    through unchanged alongside the evolving `v`, so every layer's own
#    `DataFormatter` can independently decide which of `(x,a(x),v)` it needs.
# 3. `projection` maps the final `v` alone to the output dimension (Kovachki et
#    al.'s `Q`).
# `KernelNeuralModel` itself can't be this layer directly, since it's a `NeuralModel`
# (`<: Map`, Gridap's abstract type), not an `AbstractLuxLayer`.
# """
# struct KernelOperator{N,A<:AbstractLuxLayer,B<:KernelNeuralLayer,C<:AbstractLuxLayer} <: AbstractLuxLayer
#   lifting::A
#   kernel_layers::NTuple{N,B}
#   projection::C
# end

# function Lux.initialparameters(rng::AbstractRNG,l::KernelOperator{N}) where N
#   return (
#     lifting = Lux.initialparameters(rng,l.lifting),
#     kernel_layers = ntuple(i -> Lux.initialparameters(rng,l.kernel_layers[i]),Val(N)),
#     projection = Lux.initialparameters(rng,l.projection),
#   )
# end

# function Lux.initialstates(rng::AbstractRNG,l::KernelOperator{N}) where N
#   return (
#     lifting = Lux.initialstates(rng,l.lifting),
#     kernel_layers = ntuple(i -> Lux.initialstates(rng,l.kernel_layers[i]),Val(N)),
#     projection = Lux.initialstates(rng,l.projection),
#   )
# end

# function (l::KernelOperator{N})(fixed_data,ps,st) where N
#   v,st_lift = l.lifting(fixed_data,ps.lifting,st.lifting)
#   st_layers = ()
#   for i in 1:N
#     (_,v),sti = l.kernel_layers[i]((fixed_data,v),ps.kernel_layers[i],st.kernel_layers[i])
#     st_layers = (st_layers...,sti)
#   end
#   y,st_proj = l.projection(v,ps.projection,st.projection)
#   return y,(lifting=st_lift,kernel_layers=st_layers,projection=st_proj)
# end

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