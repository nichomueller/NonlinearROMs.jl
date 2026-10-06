"""
    abstract type IntegralKernel end

Abstract supertype for integral kernel implementations used within the iterative layers.
"""
abstract type IntegralKernel end

input_size(l::IntegralKernel) = @abstractmethod
output_size(l::IntegralKernel) = @abstractmethod
input_length(l::IntegralKernel) = prod(input_size(l))
output_length(l::IntegralKernel) = prod(output_size(l))

function Lux.initialstates(rng::Random.AbstractRNG,l::IntegralKernel)
  return (
    kernel = NamedTuple(),
  )
end

struct FourierLayer{M,N} <: IntegralKernel
  shape::Dims{M}
  kmax::Dims{N}
  function FourierLayer(shape::Dims{M},kmax::Dims{N}) where {M,N}
    @check N == M-1 "FourierLayer expects kmax to have the same number of dimensions as the input shape minus 1."
    new{M,N}(shape,kmax)
  end
end

input_size(l::FourierLayer) = l.shape
output_size(l::FourierLayer) = (l.kmax...,nchannels(l))
nchannels(l::FourierLayer) = last(l.shape)

function FourierLayer(shape::Dims{M};kmax=12) where M
  FourierLayer(shape,ntuple(_->kmax,M-1))
end

function FourierLayer(n::Int,nchannels::Int;kwargs...)
  FourierLayer((n,nchannels);kwargs...)
end

function Lux.initialparameters(rng::Random.AbstractRNG,l::FourierLayer)
  return (
    kernel = rand(rng,Uniform(-1,1),l.kmax...,nchannels(l),nchannels(l)),
  )
end

function (l::FourierLayer)(x,ps,st)
  xt = truncate(x,l.kmax)
  x̂ = rfft(xt)
  ŷ = fapply(ps.kernel,x̂)
  y = irfft(ŷ)
  return y,st
end

struct GraphLayer{A<:NeuralModel,B<:WeightedSimpleDiGraph} <: IntegralKernel
  model::A
  graph::B
end

function Lux.initialparameters(rng::Random.AbstractRNG,l::GraphLayer)
  return (
    model = initialparameters(rng,l.model),
  )
end

function Lux.initialstates(rng::Random.AbstractRNG,l::GraphLayer)
  return (
    model = initialstates(rng,l.model),
  )
end

function (l::GraphLayer)(x,ps,st)
  y = zeros(size(x))
  for s in vertices(l.graph)
    xs = x[s]
    ws = out_weights(l.graph,s)
    for w in ws
      y[s] += l.model(w,ps.model,st.model)*xs
    end
  end
  return y,st
end

struct HAMLETLayer <: IntegralKernel
  # fields 
end

"""
    struct NeuralLayer{A<:IntegralKernel,B,C,D} <: Lux.AbstractLuxLayer
      kernel::A
      weights::B
      bias::C
      activation::D
    end

A single iterative layer of a Kernel Neural Operator.
It computes the update: vₜ₊₁(x) = σ(Wₜ vₜ(x) + (Kₜ vₜ)(x) + bₜ(x)), where:
- `kernel` (Kₜ) acts as the non-local integral operator.
- `weights` (Wₜ) is a pointwise local linear transformation.
- `bias` represents the dimensions of the learnable pointwise bias.
- `activation` (σ) is a fixed pointwise non-linearity.
"""
struct NeuralLayer{A<:IntegralKernel,B<:Broadcasting} <: Lux.AbstractLuxLayer
  kernel::A
  activation::B
end

function NeuralLayer(kernel::IntegralKernel,activation::Function)
  NeuralLayer(kernel,Broadcasting(activation))
end

function Lux.initialparameters(rng::Random.AbstractRNG,l::NeuralLayer)
  return (
    kernel = Lux.initialparameters(rng,l.kernel),
    weights = rand(rng,input_length(l.kernel),output_length(l.kernel)),
    bias = rand(rng,output_length(l.kernel))
  )
end

function Lux.initialstates(rng::Random.AbstractRNG,l::NeuralLayer)
  return (
    kernel = Lux.initialstates(rng,l.kernel),
  )
end

function (layer::NeuralLayer)(x,ps,st)
  # Non-local integration via specific kernel dispatch
  kout,kst = layer.kernel(x,ps.kernel,st.kernel)

  # Local linear transformation, bias addition, and activation
  wout = lapply(ps.weights,x)
  for i in eachindex(wout)
    wout[i] += kout[i] + ps.bias[i]
  end
  y = layer.activation(wout)

  return y,(kernel=kst,)
end

"""
    struct LatentCodeLayer{A<:AbstractMatrix} <: Lux.AbstractLuxLayer
      init_codes::A
    end

A Lux layer with no real input: it ignores whatever it is called with and returns its
`(latent_dim,n_train)` parameter matrix unchanged, so the per-sample latent codes of an
[`AutoDecoder`](@ref) are optimised as ordinary Lux parameters jointly with the decoder.
"""
struct LatentCodeLayer{A<:AbstractMatrix} <: Lux.AbstractLuxLayer
  init_codes::A
end

Lux.initialparameters(rng::Random.AbstractRNG,l::LatentCodeLayer) = (codes=copy(l.init_codes),)
Lux.initialstates(rng::Random.AbstractRNG,l::LatentCodeLayer) = NamedTuple()

(l::LatentCodeLayer)(x,ps,st) = ps.codes,st

struct VAELayer{E,D} <: Lux.AbstractLuxContainerLayer{(:encoder,:decoder)}
  encoder::E
  decoder::D
  latent_dim::Int
end

function (m::VAELayer)(x,ps,st)
  enc_out,st_enc = m.encoder(x,ps.encoder,st.encoder)
  μ = enc_out[1:m.latent_dim,:]
  log_var = enc_out[m.latent_dim+1:end,:]
  ε = randn(eltype(μ),size(μ))
  z = μ .+ ε .* exp.(log_var ./ 2)
  x̂,st_dec = m.decoder(z,ps.decoder,st.decoder)
  out = vcat(x̂,μ,log_var)
  return out,(encoder=st_enc,decoder=st_dec)
end

function vae_loss(model::VAELayer,ps,st,x;β=1.0)
  n = size(x,1)
  out,st = model(x,ps,st)
  x̂ = view(out,1:n,:)
  μ = view(out,n+1:n+model.latent_dim,:)
  log_var = view(out,n+model.latent_dim+1:size(out,1),:)
  recon = sum(abs2,x̂ .- x)/length(x)
  kl = -sum(1 .+ log_var .- μ.^2 .- exp.(log_var))/(2*size(x,2))
  return recon + β*kl,st,(;)
end

# utils

truncate(args...) = @notimplemented "Sizes do not match"

function truncate(x::AbstractArray{T,N},kmax::Dims{N}) where {T,N}
  s = size(x)
  s == kmax && return x
  view(x,ntuple(i->1:kmax[i],N-1)...,:)
end

lapply(W,x) = W*reshape(x,size(W,2),:)

function fapply(W::AbstractArray{T,M},x::AbstractArray{S,N}) where {T,S,M,N}
  s = size(W)[1:M-2]
  n = size(W,M)
  A = reshape(permutedims(reshape(W,:,n,n),(1,3,2)),:,n)
  b = vec(x)
  reshape(A*b,s...,n)
end