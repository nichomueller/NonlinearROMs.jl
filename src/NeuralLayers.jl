"""
    abstract type AbstractIntegralKernel <: Lux.AbstractLuxLayer end

Abstract supertype for integral kernel implementations used within the iterative layers.
"""
abstract type AbstractIntegralKernel <: Lux.AbstractLuxLayer end

"""
    struct NeuralOperatorLayer{K<:AbstractIntegralKernel, L, B, F} <: Lux.AbstractLuxContainerLayer{(:kernel, :local_linear)}

A single iterative layer of a Kernel Neural Operator.
It computes the update: vₜ₊₁(x) = σ(Wₜ vₜ(x) + (Kₜ vₜ)(x) + bₜ(x)), where:
- `kernel` (Kₜ) acts as the non-local integral operator.
- `local_linear` (Wₜ) is a pointwise local linear transformation.
- `bias_shape` represents the dimensions of the learnable pointwise bias.
- `activation` (σ) is a fixed pointwise non-linearity.
"""
struct NeuralOperatorLayer{K<:AbstractIntegralKernel,L,S,F} <: Lux.AbstractLuxLayer
  kernel::K
  local_linear::L
  bias_shape::S
  activation::F
end

# Initialize trainable parameters (ps)
function Lux.initialparameters(rng::Random.AbstractRNG,layer::NeuralOperatorLayer)
  return (
    kernel = Lux.initialparameters(rng,layer.kernel),
    local_linear = Lux.initialparameters(rng,layer.local_linear),
    # Initialize bias as zeros with the specified shape
    bias = zeros(Float32,layer.bias_shape...)
  )
end

# Initialize states (st)
function Lux.initialstates(rng::Random.AbstractRNG,layer::NeuralOperatorLayer)
  return (
    kernel = Lux.initialstates(rng,layer.kernel),
    local_linear = Lux.initialstates(rng,layer.local_linear)
  )
end

# Pre-calculate the total number of trainable parameters in this layer
function Lux.parameterlength(layer::NeuralOperatorLayer)
  kernel_params = Lux.parameterlength(layer.kernel)
  linear_params = Lux.parameterlength(layer.local_linear)
  bias_params = prod(layer.bias_shape) # Number of elements in the bias tensor

  return kernel_params + linear_params + bias_params
end

# Pre-calculate the total number of states in this layer
function Lux.statelength(layer::NeuralOperatorLayer)
  kernel_states = Lux.statelength(layer.kernel)
  linear_states = Lux.statelength(layer.local_linear)

  return kernel_states + linear_states
end

# Foward pass
function (layer::NeuralOperatorLayer)(x,ps,st)
  # Non-local integration via specific kernel dispatch
  k_out,st_k = layer.kernel(x,ps.kernel,st.kernel)

  # Local linear transformation
  w_out,st_w = layer.local_linear(x,ps.local_linear,st.local_linear)

  # Summation, bias addition, and activation
  out = layer.activation.(k_out .+ w_out .+ ps.bias)

  return out,(kernel=st_k,local_linear=st_w)
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