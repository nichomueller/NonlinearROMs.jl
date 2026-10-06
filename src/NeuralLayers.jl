"""
    abstract type AbstractIntegralKernel <: Lux.AbstractLuxLayer end

Abstract supertype for integral kernel implementations used within the iterative layers.
"""
abstract type AbstractIntegralKernel <: Lux.AbstractLuxLayer end

"""
    struct NeuralLayer{K<:AbstractIntegralKernel,L,S,F} <: Lux.AbstractLuxLayer
      kernel::K
      local_linear::L
      bias_shape::S
      activation::F
    end

A single iterative layer of a Kernel Neural Operator.
It computes the update: vₜ₊₁(x) = σ(Wₜ vₜ(x) + (Kₜ vₜ)(x) + bₜ(x)), where:
- `kernel` (Kₜ) acts as the non-local integral operator.
- `local_linear` (Wₜ) is a pointwise local linear transformation.
- `bias_shape` represents the dimensions of the learnable pointwise bias.
- `activation` (σ) is a fixed pointwise non-linearity.
"""
struct NeuralLayer{K<:AbstractIntegralKernel,L,S,F} <: Lux.AbstractLuxLayer
  kernel::K
  local_linear::L
  bias_shape::S
  activation::F
end

# Initialize trainable parameters (ps)
function Lux.initialparameters(rng::Random.AbstractRNG,layer::NeuralLayer)
  return (
    kernel = Lux.initialparameters(rng,layer.kernel),
    local_linear = Lux.initialparameters(rng,layer.local_linear),
    # Initialize bias as zeros with the specified shape
    bias = zeros(Float32,layer.bias_shape...)
  )
end

# Initialize states (st)
function Lux.initialstates(rng::Random.AbstractRNG,layer::NeuralLayer)
  return (
    kernel = Lux.initialstates(rng,layer.kernel),
    local_linear = Lux.initialstates(rng,layer.local_linear)
  )
end

# Pre-calculate the total number of trainable parameters in this layer
function Lux.parameterlength(layer::NeuralLayer)
  kernel_params = Lux.parameterlength(layer.kernel)
  linear_params = Lux.parameterlength(layer.local_linear)
  bias_params = prod(layer.bias_shape) # Number of elements in the bias tensor

  return kernel_params + linear_params + bias_params
end

# Pre-calculate the total number of states in this layer
function Lux.statelength(layer::NeuralLayer)
  kernel_states = Lux.statelength(layer.kernel)
  linear_states = Lux.statelength(layer.local_linear)

  return kernel_states + linear_states
end

struct GraphData{V,I,W}
  v::V
  edge_index::I
  edge_weights::W
end

get_features(x::AbstractArray) = x
update_features(x::AbstractArray,new_features) = new_features

get_features(x::GraphData) = x.v
update_features(x::GraphData,new_v) = GraphData(new_v,x.edge_index,x.edge_weights)

# Foward pass
function (layer::NeuralLayer)(x,ps,st)
  features = get_features(x)
  
  # Non-local integration via specific kernel dispatch
  k_out,st_k = layer.kernel(x,ps.kernel,st.kernel)

  # Local linear transformation
  w_out,st_w = layer.local_linear(features,ps.local_linear,st.local_linear)

  # Summation, bias addition, and activation
  out_data = layer.activation.(k_out .+ w_out .+ ps.bias)
  
  out = update_features(x,out_data)

  return out,(kernel=st_k,local_linear=st_w)
end

struct GNOKernel{K} <: AbstractIntegralKernel
  kernel_net::K
end

function Lux.initialparameters(rng::Random.AbstractRNG,layer::GNOKernel)
  return (kernel_net = Lux.initialparameters(rng,layer.kernel_net),)
end

function Lux.initialstates(rng::Random.AbstractRNG,layer::GNOKernel)
  return (kernel_net = Lux.initialstates(rng,layer.kernel_net),)
end

# Kernel forward pass
function (layer::GNOKernel)(x::GraphData,ps,st)
  num_features,num_nodes,batch_size = size(x.v)
  num_edges = size(x.edge_index,2)
  
  # Flatten nodes to 2D -> Shape: (C, N*B)
  v_2d = reshape(x.v,num_features,num_nodes * batch_size)
  
  # Extract 1D connectivity indices -> Shape: (E,)
  senders = vec(x.edge_index[1:1,:])
  receivers = vec(x.edge_index[2:2,:])
  
  # Take a slice of x.v of shape (1, 1, B), zero it out, and add CPU constants.
  # This forces the result to safely broadcast and live on the XLA device.
  shifts_cpu = reshape(Float32.(collect(0:(batch_size - 1)) .* num_nodes),1,1,batch_size)
  shifts_float = (x.v[1:1,1:1,:] .* 0f0) .+ shifts_cpu
  
  # Reshape to (1, B) and cast safely to Int for indexing
  shifts = round.(Int,reshape(shifts_float,1,batch_size))
  
  # Shift indices for disjoint graph and flatten -> Shape: (E*B,)
  senders_shifted = reshape(senders,num_edges,1) .+ shifts
  receivers_shifted = reshape(receivers,num_edges,1) .+ shifts
  
  # Adding + 0 to bypass reactant MethodError on ReshapedArrays
  senders_flat = reshape(senders_shifted,num_edges * batch_size) .+ 0
  receivers_flat = reshape(receivers_shifted,num_edges * batch_size) .+ 0
  
  # Gather source and target features natively in 2D -> Shape: (C, E*B)
  source_features_2d = v_2d[:,senders_flat]
  target_features_2d = v_2d[:,receivers_flat]
  
  # Broadcast edge weights to match batch dimension -> Shape: (1, E*B)
  weights_rep = repeat(reshape(x.edge_weights,1,num_edges),1,batch_size)
  
  # Add + 0f0 to force contiguous array
  weights_flat = reshape(weights_rep,1,num_edges * batch_size) .+ 0f0
  
  # Concatenate features for the MLP -> Shape: (2C+1, E*B)
  edge_features_2d = vcat(source_features_2d,target_features_2d,weights_flat)
  
  # Compute edge message weights (Kappa) via MLP -> Shape: (F_out, E*B)
  kappa_2d,updated_st = layer.kernel_net(edge_features_2d,ps.kernel_net,st.kernel_net)
  
  # Modulate source features with computed weights -> Shape: (F_out, E*B)
  messages_2d = kappa_2d .* source_features_2d
  
  # Add + 0f0 to force contiguous array before scatter
  messages_2d_safe = messages_2d .+ 0f0
  
  # Scatter-add to aggregate messages at destination nodes -> Shape: (F_out, N*B)
  out_features = size(messages_2d_safe,1)
  aggregated_2d = Lux.NNlib.scatter(+,messages_2d_safe,receivers_flat; dstsize=(out_features,num_nodes * batch_size))
  
  # Degree normalization (Tracing and AD safe pseudo-allocation)
  ones_flat = (weights_flat .* 0f0) .+ 1f0
  node_degrees_1d = Lux.NNlib.scatter(+,ones_flat,receivers_flat; dstsize=(1,num_nodes * batch_size))
  safe_node_degrees_1d = max.(node_degrees_1d,1f0)
  
  # Average aggregated messages -> Shape: (F_out, N*B)
  normalized_2d = aggregated_2d ./ safe_node_degrees_1d
  
  # Reshape back to 3D -> Shape: (F_out, N, B)
  normalized_messages = reshape(normalized_2d,out_features,num_nodes,batch_size)
  
  return normalized_messages,(kernel_net=updated_st,)
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