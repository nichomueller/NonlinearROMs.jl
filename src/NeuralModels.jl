"""
    abstract type NeuralModel <: Map end

Abstract supertype for neural network models. Any concrete subtype must
implement `(a::T)(x::AbstractMatrix) -> AbstractMatrix` where `x` is a
`(param_dim × batch_size)` matrix of parameters and the output is a
`(output_dim × batch_size)` matrix of predictions.
"""
abstract type NeuralModel <: Map end

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
    struct NOMAD{N,A} <: NeuralModel
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
struct NOMAD{N,A} <: NeuralModel
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
    struct MultiLayerPerceptron{N,A} <: NeuralModel
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
struct MultiLayerPerceptron{N,A} <: NeuralModel
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

const CDEV = Lux.cpu_device()
const XDEV = Lux.reactant_device(;force=true)

"""
    struct TrainedNeuralModel{A,B,C,D} <: NeuralModel
      chain::A
      parameters::B
      states::C
      stats::D
    end

A trained Lux `chain` bundled with its optimised parameters/states and the
normalisation `stats` (a [`Normalisation`](@ref)/[`ZScore`](@ref), or `identity`
if none) used to pre-normalise inputs and de-normalise outputs. Evaluable as
`(a::TrainedNeuralModel)(x) -> AbstractMatrix`.
"""
struct TrainedNeuralModel{A,B,C,D} <: NeuralModel
  chain::A
  parameters::B
  states::C
  stats::D
end

function TrainedNeuralModel(chain,parameters,states)
  TrainedNeuralModel(chain,parameters,states,identity)
end

function TrainedNeuralModel(train_state::Lux.Training.TrainState,args...)
  parameters = train_state.parameters |> CDEV
  states = Lux.testmode(train_state.states) |> CDEV
  TrainedNeuralModel(train_state.model,parameters,states,args...)
end

"""
    (m::TrainedNeuralModel)(inputs) -> AbstractArray

Applies `m` to `inputs` (a `(params,coords)`/`(pin,xin)` tuple for DeepONet/NOMAD, or
a plain matrix): normalises `inputs` in place with `m.stats`, runs the Lux chain, then
rescales the prediction back with `m.stats`. Both steps are no-ops when `m.stats === identity`
(a model with no normalisation, e.g. an `AutoEncoder`/`AutoDecoder`/`VAE`).
"""
function (m::TrainedNeuralModel)(inputs)
  normalise!(inputs,m.stats)
  pred = first(m.chain(inputs,m.parameters,m.states))
  denormalise!(pred,m.stats)
  return pred
end