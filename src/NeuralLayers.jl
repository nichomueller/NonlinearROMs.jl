"""
    abstract type IntegralKernel end

Abstract supertype for integral kernel implementations used within the iterative layers.
"""
abstract type IntegralKernel end

struct FourierLayer <: IntegralKernel 
  # fields 
end

struct GraphLayer <: IntegralKernel 
  # fields 
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
struct NeuralLayer{A<:IntegralKernel,B,C,D} <: Lux.AbstractLuxLayer
  kernel::A
  weights::B
  bias::C
  activation::D
end

# Initialize trainable parameters (ps)
function Lux.initialparameters(rng::Random.AbstractRNG,layer::NeuralLayer)
  return (
    kernel = Lux.initialparameters(rng,layer.kernel),
    weights = Lux.initialparameters(rng,layer.weights),
    bias = Lux.initialparameters(rng,layer.bias)
  )
end

# Initialize states (st)
function Lux.initialstates(rng::Random.AbstractRNG,layer::NeuralLayer)
  return (
    kernel = Lux.initialstates(rng,layer.kernel),
    weights = Lux.initialstates(rng,layer.weights)
  )
end

# Pre-calculate the total number of trainable parameters in this layer
function Lux.parameterlength(layer::NeuralLayer)
  kernel_params = Lux.parameterlength(layer.kernel)
  linear_params = Lux.parameterlength(layer.weights)
  bias_params = Lux.parameterlength(layer.bias)
  return kernel_params + linear_params + bias_params
end

# Pre-calculate the total number of states in this layer
function Lux.statelength(layer::NeuralLayer)
  kernel_states = Lux.statelength(layer.kernel)
  linear_states = Lux.statelength(layer.weights)
  return kernel_states + linear_states
end

# Foward pass
function (layer::NeuralLayer)(x,ps,st)
  # Non-local integration via specific kernel dispatch
  k_out,st_k = layer.kernel(x,ps.kernel,st.kernel)

  # Local linear transformation
  w_out,st_w = layer.weights(x,ps.weights,st.weights)

  # Summation, bias addition, and activation
  out = layer.activation.(k_out .+ w_out .+ ps.bias)

  return out,(kernel=st_k,weights=st_w)
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