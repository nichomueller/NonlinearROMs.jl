"""
    abstract type NeuralModel <: Map end

Abstract supertype for neural network models. Any concrete subtype must
implement `(a::T)(x::AbstractMatrix) -> AbstractMatrix` where `x` is a
`(param_dim × batch_size)` matrix of parameters and the output is a
`(output_dim × batch_size)` matrix of predictions.
"""
abstract type NeuralModel <: Map end

"""
    abstract type CoordinateNeuralModel <: NeuralModel end

Abstract supertype for neural models that evaluate the solution point-by-point using continuous physical coordinates.
"""
abstract type CoordinateNeuralModel <: NeuralModel end

"""
    struct KernelNeuralModel{N,A,B,C,D} <: NeuralModel
      lifting::A
      kernel_layers::NTuple{N,B}
      projection::C
      activation::D
    end

Generic architecture for kernel-based neural models.
It maps an input function to an output function through three main stages:
1. Lifting (P): A local operator mapping input features to a higher-dimensional hidden representation.
2. Iterative Kernel Integration: A sequence of `KernelNeuralLayer`s representing the non-local processing.
3. Projection (Q): A local operator mapping the final hidden representation to the target output dimension.
"""
struct KernelNeuralModel{N,A,B,C,D} <: NeuralModel
  lifting::A
  kernel_layers::NTuple{N,B}
  projection::C
  activation::D
end

"""
    struct DeepONet{N,A} <: CoordinateNeuralModel
      branch_layers::NTuple{N,Int}
      trunk_layers::NTuple{N,Int}
      activation::A
    end

Explicit architectural configuration for a Deep Operator Model (DeepONet).
A DeepONet consists of two sub-networks:
1. **Branch Net**: Processes the input parameters/sensors \$u\$.
2. **Trunk Net**: Processes the spatial/spatiotemporal continuous coordinates \$y\$.

The final output is computed via the dot product of their feature vectors:
\$\$G(u)(y) = \\sum_{k=1}^{p} b_k(u) t_k(y)\$\$

The last layer of both `branch_layers` and `trunk_layers` must have the exact same dimension \$p\$.

# Examples
The layers are defined as standard Julia tuples.
```julia
# Branch Net: 2 inputs -> 64 hidden -> 32 output (latent dimension p=32)
# Trunk Net: 3 inputs (e.g., 2D space + time) -> 64 hidden -> 32 output
model = DeepONet(
  branch_layers = (2,64,32),
  trunk_layers = (3,64,32),
  activation = relu
  )
```
"""
struct DeepONet{N,A} <: CoordinateNeuralModel
  branch_layers::NTuple{N,Int}
  trunk_layers::NTuple{N,Int}
  activation::A
end

function DeepONet(;branch_layers,trunk_layers,activation=tanh)
  DeepONet(Tuple(branch_layers),Tuple(trunk_layers),activation)
end

function DeepONet(
  nbranch_in::Int,
  ntrunk_in::Int;
  width::Int=64,
  depth::Int=3,
  hidden=ntuple(_ -> width,depth),
  branch_layers=(nbranch_in,hidden...,width),
  trunk_layers=(ntrunk_in,hidden...,width),
  activation=tanh
  )

  DeepONet(Tuple(branch_layers),Tuple(trunk_layers),activation)
end

"""
    struct NOMAD{N,A} <: CoordinateNeuralModel
      approximator_layers::NTuple{N,Int}
      decoder_layers::NTuple{N,Int}
      activation::A
    end

Explicit architectural configuration for a NOMAD (Non-linear Manifold Decoder) network.
It uses an Approximator (Encoder) to map the parametric sensors into a latent space, and a Decoder that takes the concatenated vector of the latent representation and the physical coordinates to predict the solution field.

# Examples
The layers are defined as standard Julia tuples.
```julia
# Approximator: 5 sensors -> 32 hidden -> 16 latent space
# Decoder: 19 inputs (16 latent + 3 spatial coords) -> 32 hidden -> 1 output
model = NOMAD(
  approximator_layers = (5,32,16),
  decoder_layers = (19,32,1),
  activation = relu
  )
```
"""
struct NOMAD{N,A} <: CoordinateNeuralModel
  approximator_layers::NTuple{N,Int}
  decoder_layers::NTuple{N,Int}
  activation::A
end

function NOMAD(;approximator_layers,decoder_layers,activation=tanh)
  NOMAD(Tuple(approximator_layers),Tuple(decoder_layers),activation)
end

function NOMAD(
  nsensors_in::Int,
  ncoords_in::Int;
  width::Int=64,
  depth::Int=3,
  hidden=ntuple(_ -> width,depth),
  approximator_layers=(nsensors_in,hidden...,width),
  decoder_layers=(width+ncoords_in,hidden...,1),
  activation=tanh
  )

  NOMAD(Tuple(approximator_layers),Tuple(decoder_layers),activation)
end

"""
    abstract type FiniteDimensionalModel <: NeuralModel end

Abstract supertype for standard neural networks mapping between finite-dimensional Euclidean spaces (Vector -> Vector).
"""
abstract type FiniteDimensionalModel <: NeuralModel end

"""
    struct MultiLayerPerceptron{N,A} <: FiniteDimensionalModel
      hidden_layers::NTuple{N,Int}
      activation::A
    end

Recipe for a dense feed-forward network used for scalar/vector regression (e.g.
predicting EIM/reduced-basis coefficients from parameter values). `hidden_layers`
holds only the hidden widths (e.g. `(64,64)`); the input/output dimensions are
inferred from the training data at [`train_neural_coefficient`](@ref) call time, since
one `MultiLayerPerceptron` recipe is typically reused (via [`NeuralReduction`](@ref))
to train many differently-shaped networks (one per triangulation/reduced quantity).
"""
struct MultiLayerPerceptron{N,A} <: FiniteDimensionalModel
  hidden_layers::NTuple{N,Int}
  activation::A
end

function MultiLayerPerceptron(;
  width::Int=64,
  depth::Int=3,
  hidden=ntuple(_ -> width,depth),
  activation=tanh
  )

  MultiLayerPerceptron(Tuple(hidden),activation)
end

"""
    struct AutoEncoder{N,A} <: FiniteDimensionalModel
      hidden_layers::NTuple{N,Int}
      activation::A
    end

Recipe for an encoder-decoder pair for unsupervised dimensionality reduction.
`hidden_layers = (h₁,…,h_{L-1},latent_dim)`: the encoder hidden widths are
`(h₁,…,h_{L-1})` and the decoder mirrors them symmetrically; the input dimension
is inferred from the training data.
"""
struct AutoEncoder{N,A} <: FiniteDimensionalModel
  hidden_layers::NTuple{N,Int}
  activation::A
end

function AutoEncoder(
  width::Int=64,
  depth::Int=3,
  hidden=ntuple(_ -> width,depth),
  activation=tanh
  )

  AutoEncoder(Tuple(hidden),activation)
end

"""
    struct VariationalAutoEncoder{N,A,B} <: FiniteDimensionalModel
      hidden_layers::NTuple{N,Int}
      activation::A
      β::Float64
    end

Recipe for a VAE with the reparameterisation trick. `hidden_layers = (h₁,…,h_{L-1},latent_dim)`,
interpreted as for [`AutoEncoder`](@ref); `β` weighs the KL term against the
reconstruction loss.
"""
struct VariationalAutoEncoder{N,A,B} <: FiniteDimensionalModel
  hidden_layers::NTuple{N,Int}
  activation::A
  β::B
end

function VariationalAutoEncoder(
  width::Int=64,
  depth::Int=3,
  hidden=ntuple(_ -> width,depth),
  activation=tanh,
  β=1.0
  )

  VariationalAutoEncoder(Tuple(hidden),activation,Float64(β))
end

"""
    struct AutoDecoder{N,A} <: FiniteDimensionalModel
      hidden_layers::NTuple{N,Int}
      activation::A
    end

Recipe for a decoder-only model (Park et al., 2019). Per-sample latent codes are
optimised jointly with the decoder parameters. `hidden_layers = (h₁,…,latent_dim)`;
the decoder is built from last to first, i.e. `(latent_dim,reverse(h₁,…,h_{L-1})…,n_h)`.
"""
struct AutoDecoder{N,A} <: FiniteDimensionalModel
  hidden_layers::NTuple{N,Int}
  activation::A
end

function AutoDecoder(width::Int=64,
  depth::Int=3,
  hidden=ntuple(_ -> width,depth),
  activation=tanh
  )

  AutoDecoder(Tuple(hidden),activation)
end

# Build model

build_model(::NeuralModel,args...) = @abstractmethod

function build_lux_chain(layers::Tuple,activation)
  lux_layers = []
  for i in 1:(length(layers)-1)
    if i < length(layers)-1
      push!(lux_layers,Lux.Dense(layers[i] => layers[i+1],activation))
    else
      # last layer (no activation)
      push!(lux_layers,Lux.Dense(layers[i] => layers[i+1]))
    end
  end
  Lux.Chain(lux_layers...)
end

# Create a DeepONet layers
function LuxDeepONet(branch_net,trunk_net)
  Lux.Chain(
    # Process inputs (u,y) independently,then matrix-multiply them
    Lux.Parallel(
      *;
      # Branch: process 'u' -> shape (Features,Batch)
      # then transpose (adjoint) -> shape (Batch,Features)
      branch = Lux.Chain(branch_net,Lux.WrappedFunction(adjoint)),

      # Trunk: process 'y' -> shape (Features,Points)
      trunk = trunk_net
    ),
    # The '*' gives (Batch,Points).
    # Final transpose (adjoint) -> target shape: (Points,Batch)
    Lux.WrappedFunction(adjoint)
  )
end

function build_model(model::DeepONet,args...)
  branch_net = build_lux_chain(model.branch_layers,model.activation)
  trunk_net = build_lux_chain(model.trunk_layers,model.activation)
  LuxDeepONet(branch_net,trunk_net)
end

function LuxNOMAD(approximator_net,decoder_net)
  Lux.Chain(
    # Apply approximator to 'u',pass 'y' untouched,and concatenate them (vcat)
    Lux.Parallel(
      vcat;
      approximator = approximator_net,
      y_pass_through = Lux.NoOpLayer()
    ),
    # Pass the concatenated vector [approximator(u); y] to the decoder
    decoder_net
  )
end

function build_model(model::NOMAD,args...)
  approximator_net = build_lux_chain(model.approximator_layers,model.activation)
  decoder_net = build_lux_chain(model.decoder_layers,model.activation)
  LuxNOMAD(approximator_net,decoder_net)
end

function build_model(model::AutoEncoder,(values,coords,params))
  n = size(values,1)
  hidden = model.hidden_layers[1:end-1]
  latent_dim = last(model.hidden_layers)
  encoder = build_lux_chain((n,hidden...,latent_dim),model.activation)
  decoder = build_lux_chain((latent_dim,reverse(hidden)...,n),model.activation)
  Lux.Chain(encoder,decoder)
end

function build_model(model::AutoDecoder,(values,coords,params))
  n,ntrain = size(values,1),size(values,2)
  hidden = model.hidden_layers[1:end-1]
  latent_dim = last(model.hidden_layers)
  decoder = build_lux_chain((latent_dim,reverse(hidden)...,n),model.activation)
  Z0 = randn(Float32,latent_dim,ntrain) .* 0.01f0
  Lux.Chain(LatentCodeLayer(Z0),decoder)
end

function build_model(model::VariationalAutoEncoder,(values,coords,params))
  n = size(values,1)
  hidden = model.hidden_layers[1:end-1]
  latent_dim = last(model.hidden_layers)
  encoder = build_lux_chain((n,hidden...,2*latent_dim),model.activation)
  decoder = build_lux_chain((latent_dim,reverse(hidden)...,n),model.activation)
  VAELayer(encoder,decoder,latent_dim)
end

function build_model(model::MultiLayerPerceptron,(values,coords,params))
  m = size(params,1)
  n = size(values,1)
  build_lux_chain((m,model.hidden_layers...,n),model.activation)
end