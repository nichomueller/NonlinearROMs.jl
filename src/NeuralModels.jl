"""
    abstract type NeuralModel end

Abstract supertype for neural network models. Any concrete subtype must
implement `(a::T)(x::AbstractMatrix) -> AbstractMatrix` where `x` is a
`(param_dim × batch_size)` matrix of parameters and the output is a
`(output_dim × batch_size)` matrix of predictions.
"""
abstract type NeuralModel end

DataFormatter(::NeuralModel) = DataFormatter()

abstract type KernelStyle end

DataFormatter(::KernelStyle) = DataFormatter()

struct IntegralStyle{A<:InducedMeasure} <: KernelStyle 
  measure::A
end

struct FourierStyle <: KernelStyle 
  truncation::Int
end

function DataFormatter(style::FourierStyle)
  DataFormatter(pass_coords=Val{false}(),pass_features=Val{false}(),pass_new_data=Val{true}())
end

"""
    struct KernelNeuralModel{A<:KernelStyle,N,B<:Function,C<:DataFormatter} <: NeuralModel
      style::A
      hidden_layers::NTuple{N,Int}
      activation::B
      formatter::C
    end

Recipe for a kernel-based neural operator (Kovachki et al.), in the same spirit as
[`NOMAD`](@ref)/[`MultiLayerPerceptron`](@ref): `hidden_layers` only holds the
hidden widths, with the input/output dimensions inferred from the training data
at `build_model` time. `style` picks the non-local discretisation shared by every
hidden layer (`IntegralStyle` for a graph/FE-quadrature [`InducedMeasure`](@ref),
`FourierStyle` for an FNO-style spectral kernel); `formatter` is the single
[`DataFormatter`](@ref) that both decides what the model's fixed per-sample input
is made of (`x`/`a(x)`) and, inside every hidden layer, whether the running hidden
state `v` gets folded into that kernel's own input.

`build_model` turns this recipe into an actual `Lux.Chain`:
1. Lifting (P): a local `Dense` layer mapping the fixed input to the first hidden width.
2. Iterative Kernel Integration: a stack of `KernelNeuralLayer`s (one per interior
   hidden-width transition) performing the non-local processing.
3. Projection (Q): a local `Dense` layer mapping the final hidden state to the output dimension.
"""
struct KernelNeuralModel{A<:KernelStyle,N,B<:Function,C<:DataFormatter} <: NeuralModel
  style::A
  hidden_layers::NTuple{N,Int}
  activation::B
  formatter::C
end

function KernelNeuralModel(
  style::KernelStyle;
  hidden_layers=(16,16),
  activation=tanh,
  formatter=DataFormatter(style)
  )

  KernelNeuralModel(style,hidden_layers,activation,formatter)
end

function KernelNeuralModel(f::DiscreteSpace,args...;kwargs...)
  style = IntegralStyle(InducedMeasure(f))
  KernelNeuralModel(style,args...;kwargs...)
end

function KernelNeuralModel(args...;truncation=16,kwargs...)
  style = FourierStyle(truncation)
  KernelNeuralModel(style,args...;kwargs...)
end

DataFormatter(model::KernelNeuralModel) = model.formatter

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
  hidden_layers=ntuple(_ -> width,depth),
  activation=tanh
  )

  MultiLayerPerceptron(hidden_layers,activation)
end