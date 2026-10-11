"""
    abstract type IntegralKernel <: Lux.AbstractLuxLayer end

Abstract supertype for integral kernel implementations used within the iterative layers.
"""
abstract type IntegralKernel <: Lux.AbstractLuxLayer end

input_size(l::IntegralKernel) = @abstractmethod
output_size(l::IntegralKernel) = @abstractmethod
input_length(l::IntegralKernel) = prod(input_size(l))
output_length(l::IntegralKernel) = prod(output_size(l))

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

function Lux.initialstates(rng::Random.AbstractRNG,l::FourierLayer)
  return (
    kernel = NamedTuple(),
  )
end

function (l::FourierLayer)(x,ps,st)
  xt = truncate(x,l.kmax)
  x̂ = rfft(xt)
  ŷ = fapply(ps.kernel,x̂)
  y = irfft(ŷ)
  return y,st
end

abstract type GraphLayerStyle end
struct FullGraph <: GraphLayerStyle end
struct NystromGraph <: GraphLayerStyle 
  num_subgraphs::Int
  subgraph_size::Int
end
NystromGraph(;num_subgraphs=4,subgraph_size=100) = NystromGraph(num_subgraphs,subgraph_size)

struct GraphLayer{A<:GraphLayerStyle,B<:NeuralModel,C<:WeightedSimpleDiGraph} <: IntegralKernel
  style::A
  model::B
  graph::C
end

const FullGraphLayer{B<:NeuralModel,C<:WeightedSimpleDiGraph} = GraphLayer{FullGraph,B,C}
const NystromGraphLayer{B<:NeuralModel,C<:WeightedSimpleDiGraph} = GraphLayer{NystromGraph,B,C}

input_size(l::GraphLayer) = input_size(l.model)
output_size(l::GraphLayer) = output_size(l.model)

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

function (l::FullGraphLayer)(x,ps,st)
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

function (l::NystromGraphLayer)(x,ps,st)
  y = zeros(size(x))
  for _ in 1:l.style.num_subgraphs
    subgraph,vmap = sample_subgraph(l.graph,l.style.subgraph_size)
    for ss in vertices(subgraph)
      s = vmap[ss]
      xs = x[s]
      ws = out_weights(subgraph,ss)
      for w in ws
        y[s] += l.model(w,ps.model,st.model)*xs
      end
    end
  end
  return y,st
end

abstract type AttentionModel <: IntegralKernel end

function Lux.initialstates(rng::Random.AbstractRNG,l::AttentionModel)
  return (
    kernel = NamedTuple(),
  )
end

struct SingleHeadAttention <: AttentionModel
  nkeys::Int
  nvalues::Int
  dimension::Int
end

function SingleHeadAttention(;nkeys=10,nvalues=nkeys,dimension=100)
  SingleHeadAttention(nkeys,nvalues,dimension)
end

function Lux.initialparameters(rng::Random.AbstractRNG,l::SingleHeadAttention)
  return (
    queries = rand(rng,l.dimension,l.nkeys),
    keys = rand(rng,l.dimension,l.nkeys),
    values = rand(rng,l.dimension,l.nvalues),
  )
end

function (l::SingleHeadAttention)(x,ps,st)
  Q,K,V = attention_matrices(x,ps)
  y = softmax(Q * K' / sqrt(l.nkeys)) * V 
  return y,st
end

struct GraphSingleHeadAttention{A<:GraphLayer} <: AttentionLayer
  graph::A
  nkeys::Int
  nvalues::Int
  dimension::Int
end

function SingleHeadAttention(graph::GraphLayer;nkeys=10,nvalues=nkeys,dimension=100)
  GraphSingleHeadAttention(graph,nkeys,nvalues,dimension)
end

function Lux.initialparameters(rng::Random.AbstractRNG,l::SingleHeadAttention)
  return (
    queries = rand(rng,l.dimension,l.nkeys),
    keys = rand(rng,l.dimension,l.nkeys),
    values = rand(rng,l.dimension,l.nvalues),
  )
end

function (l::NystromGraphLayer)(x,ps,st)
  y = zeros(size(x))
  for _ in 1:l.style.num_subgraphs
    subgraph,vmap = sample_subgraph(l.graph,l.style.subgraph_size)
    for ss in vertices(subgraph)
      s = vmap[ss]
      xs = x[s]
      ws = out_weights(subgraph,ss)
      for w in ws
        y[s] += l.model(w,ps.model,st.model)*xs
      end
    end
  end
  return y,st
end

function (l::GraphSingleHeadAttention{<:FullGraphLayer})(x,ps,st)
  Q,K,V = attention_matrices(x,ps)
  y = zeros(size(x))
  for s in vertices(l.graph)
    α = 0.0
    for t in neighbors(l.graph,s)
      @views α += softmax(Q[:,s]' * K[:,t] / sqrt(l.nkeys))
    end
    @views y[s,:] += α * V[:,t]
  end
  return y,st
end

function (l::GraphSingleHeadAttention{<:NystromGraphLayer})(x,ps,st)
  Q,K,V = attention_matrices(x,ps)
  y = zeros(size(x))
  for _ in 1:l.style.num_subgraphs
    subgraph,vmap = sample_subgraph(l.graph,l.style.subgraph_size)
    for ss in vertices(subgraph)
      s = vmap[ss]
      α = 0.0
      for t in neighbors(l.graph,s)
        @views α += softmax(Q[:,s]' * K[:,t] / sqrt(l.nkeys))
      end
      @views y[s,:] += α * V[:,t]
    end
  end
  return y,st
end

struct MultiHeadAttention{A<:AttentionModel} <: AttentionModel
  attention::A
  nheads::Int
end

const GraphMultiHeadAttention = MultiHeadAttention{<:GraphSingleHeadAttention}

function MultiHeadAttention(;nheads=1,nkeys=10,nvalues=nkeys,dimension=nkeys*nheads)
  @check dimension == nkeys * nheads
  attention = SingleHeadAttention(nkeys,nvalues,dimension)
  MultiHeadAttention(attention,nheads)
end

function MultiHeadAttention(graph::GraphLayer;nheads=1,nkeys=10,nvalues=nkeys,dimension=nkeys*nheads)
  @check dimension == nkeys * nheads
  attention = SingleHeadAttention(graph,nkeys,nvalues,dimension)
  MultiHeadAttention(attention,nheads)
end

function Lux.initialparameters(rng::Random.AbstractRNG,l::MultiHeadAttention)
  return (
    heads = (Lux.initialparameters(rng,l.attention) for _ in 1:l.nheads),
  )
end

function Lux.initialstates(rng::Random.AbstractRNG,l::MultiHeadAttention)
  return (
    heads = (Lux.initialstates(rng,l.attention) for _ in 1:l.nheads),
  )
end

function (l::MultiHeadAttention)(x,ps,st)
  nk = l.attention.nkeys
  y = zeros(size(x))
  for i in 1:l.nheads
    @view y[:,(i-1)*nk+1:i*nk] = l.attention(x,ps.heads[i],st.heads[i])
  end
  return y,st
end

struct HAMLETLayer{A<:AttentionModel} <: IntegralKernel
  attention::A
end

"""
    struct KernelNeuralLayer{A<:IntegralKernel,B,C,D} <: Lux.AbstractLuxLayer
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
struct KernelNeuralLayer{A<:IntegralKernel,B<:Broadcasting} <: Lux.AbstractLuxLayer
  kernel::A
  activation::B
end

function KernelNeuralLayer(kernel::IntegralKernel,activation::Function)
  KernelNeuralLayer(kernel,Broadcasting(activation))
end

function Lux.initialparameters(rng::Random.AbstractRNG,l::KernelNeuralLayer)
  return (
    kernel = Lux.initialparameters(rng,l.kernel),
    weights = rand(rng,input_length(l.kernel),output_length(l.kernel)),
    bias = rand(rng,output_length(l.kernel))
  )
end

function Lux.initialstates(rng::Random.AbstractRNG,l::KernelNeuralLayer)
  return (
    kernel = Lux.initialstates(rng,l.kernel),
  )
end

function (layer::KernelNeuralLayer)(x,ps,st)
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

function attention_matrices(x,ps)
  Q = x * ps.queries
  K = x * ps.keys
  V = x * ps.values
  return Q,K,V
end

# self attention
function attention_matrices(x,ps)
  Q = x * ps.queries
  K = x * ps.keys
  V = x * ps.values
  return Q,K,V
end

# cross attention
function attention_matrices(x::NTuple{2},ps)
  xq,xk = x
  Q = xq * ps.queries
  K = xk * ps.keys
  V = xk * ps.values
  return Q,K,V
end

#TODO actually merge Isaia's code:

# struct GraphData{V,I,W}
#   v::V
#   edge_index::I
#   edge_weights::W
# end

# get_features(x::AbstractArray) = x
# update_features(x::AbstractArray,new_features) = new_features

# get_features(x::GraphData) = x.v
# update_features(x::GraphData,new_v) = GraphData(new_v,x.edge_index,x.edge_weights)

# # Foward pass
# function (layer::NeuralLayer)(x,ps,st)
#   features = get_features(x)
  
#   # Non-local integration via specific kernel dispatch
#   k_out,st_k = layer.kernel(x,ps.kernel,st.kernel)

#   # Local linear transformation
#   w_out,st_w = layer.local_linear(features,ps.local_linear,st.local_linear)

#   # Summation, bias addition, and activation
#   out_data = layer.activation.(k_out .+ w_out .+ ps.bias)
  
#   out = update_features(x,out_data)

#   return out,(kernel=st_k,local_linear=st_w)
# end

# struct GNOKernel{K} <: AbstractIntegralKernel
#   kernel_net::K
# end

# function Lux.initialparameters(rng::Random.AbstractRNG,layer::GNOKernel)
#   return (kernel_net = Lux.initialparameters(rng,layer.kernel_net),)
# end

# function Lux.initialstates(rng::Random.AbstractRNG,layer::GNOKernel)
#   return (kernel_net = Lux.initialstates(rng,layer.kernel_net),)
# end

# # Kernel forward pass
# function (layer::GNOKernel)(x::GraphData,ps,st)
#   num_features,num_nodes,batch_size = size(x.v)
#   num_edges = size(x.edge_index,2)
  
#   # Flatten nodes to 2D -> Shape: (C, N*B)
#   v_2d = reshape(x.v,num_features,num_nodes * batch_size)
  
#   # Extract 1D connectivity indices -> Shape: (E,)
#   senders = vec(x.edge_index[1:1,:])
#   receivers = vec(x.edge_index[2:2,:])
  
#   # Take a slice of x.v of shape (1, 1, B), zero it out, and add CPU constants.
#   # This forces the result to safely broadcast and live on the XLA device.
#   shifts_cpu = reshape(Float32.(collect(0:(batch_size - 1)) .* num_nodes),1,1,batch_size)
#   shifts_float = (x.v[1:1,1:1,:] .* 0f0) .+ shifts_cpu
  
#   # Reshape to (1, B) and cast safely to Int for indexing
#   shifts = round.(Int,reshape(shifts_float,1,batch_size))
  
#   # Shift indices for disjoint graph and flatten -> Shape: (E*B,)
#   senders_shifted = reshape(senders,num_edges,1) .+ shifts
#   receivers_shifted = reshape(receivers,num_edges,1) .+ shifts
  
#   # Adding + 0 to bypass reactant MethodError on ReshapedArrays
#   senders_flat = reshape(senders_shifted,num_edges * batch_size) .+ 0
#   receivers_flat = reshape(receivers_shifted,num_edges * batch_size) .+ 0
  
#   # Gather source and target features natively in 2D -> Shape: (C, E*B)
#   source_features_2d = v_2d[:,senders_flat]
#   target_features_2d = v_2d[:,receivers_flat]
  
#   # Broadcast edge weights to match batch dimension -> Shape: (1, E*B)
#   weights_rep = repeat(reshape(x.edge_weights,1,num_edges),1,batch_size)
  
#   # Add + 0f0 to force contiguous array
#   weights_flat = reshape(weights_rep,1,num_edges * batch_size) .+ 0f0
  
#   # Concatenate features for the MLP -> Shape: (2C+1, E*B)
#   edge_features_2d = vcat(source_features_2d,target_features_2d,weights_flat)
  
#   # Compute edge message weights (Kappa) via MLP -> Shape: (F_out, E*B)
#   kappa_2d,updated_st = layer.kernel_net(edge_features_2d,ps.kernel_net,st.kernel_net)
  
#   # Modulate source features with computed weights -> Shape: (F_out, E*B)
#   messages_2d = kappa_2d .* source_features_2d
  
#   # Add + 0f0 to force contiguous array before scatter
#   messages_2d_safe = messages_2d .+ 0f0
  
#   # Scatter-add to aggregate messages at destination nodes -> Shape: (F_out, N*B)
#   out_features = size(messages_2d_safe,1)
#   aggregated_2d = Lux.NNlib.scatter(+,messages_2d_safe,receivers_flat; dstsize=(out_features,num_nodes * batch_size))
  
#   # Degree normalization (Tracing and AD safe pseudo-allocation)
#   ones_flat = (weights_flat .* 0f0) .+ 1f0
#   node_degrees_1d = Lux.NNlib.scatter(+,ones_flat,receivers_flat; dstsize=(1,num_nodes * batch_size))
#   safe_node_degrees_1d = max.(node_degrees_1d,1f0)
  
#   # Average aggregated messages -> Shape: (F_out, N*B)
#   normalized_2d = aggregated_2d ./ safe_node_degrees_1d
  
#   # Reshape back to 3D -> Shape: (F_out, N, B)
#   normalized_messages = reshape(normalized_2d,out_features,num_nodes,batch_size)
  
#   return normalized_messages,(kernel_net=updated_st,)
# end