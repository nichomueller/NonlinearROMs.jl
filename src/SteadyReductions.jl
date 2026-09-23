abstract type AbstractNNHyperReduction{A<:ReductionStyle} <: HyperReduction{A} end

"""
    struct NNOperatorReduction <: AbstractNNHyperReduction{NoReduction}
      nparams::Int
      strategy::NeuralStrategy
    end

A hyper-reduction strategy for **operator regression**: the NN directly maps
parameter values to the Galerkin-projected residual vector, bypassing FE
assembly entirely during the online phase. Only suitable for residual
(vector-valued) operators.

The offline phase projects the residual snapshots onto the test space and
trains the NN to reproduce the projected vectors. The online phase calls
the NN forward pass, producing the projected residual without any assembly.

`strategy` controls the [`MultiLayerPerceptron`](@ref) architecture and training.
`nparams` controls how many parameter samples to use for NN training.
"""
struct NNOperatorReduction <: AbstractNNHyperReduction{NoReductionStyle}
  nparams::Int
  strategy::NeuralStrategy
end

function NNOperatorReduction(
  args...;
  nparams::Int=20,
  model::NeuralNetwork=MultiLayerPerceptron(),
  strategy::NeuralStrategy=NeuralStrategy(model),
  kwargs...
  )

  NNOperatorReduction(nparams,strategy)
end

ParamDataStructures.num_params(r::NNOperatorReduction) = r.nparams
get_strategy(r::NNOperatorReduction) = r.strategy

"""
    struct NNHyperReduction{A} <: AbstractNNHyperReduction{A}
      reduction::Reduction{A,EuclideanNorm}
      strategy::NeuralStrategy
    end

A hyper-reduction strategy that uses a neural network to predict EIM
coefficients from parameter values. The offline phase:

1. applies empirical interpolation on the snapshot basis to extract coefficients
2. trains a [`MultiLayerPerceptron`](@ref) via `strategy` on the `(μ,coefficient)` pairs

The online phase calls the NN forward pass instead of assembling the FE
operator on the reduced integration domain.
"""
struct NNHyperReduction{A} <: AbstractNNHyperReduction{A}
  reduction::Reduction{A,EuclideanNorm}
  strategy::NeuralStrategy
end

"""
    NNHyperReduction(args...;model=MultiLayerPerceptron(),strategy=NeuralStrategy(model),kwargs...) -> NNHyperReduction

Constructs a `NNHyperReduction` from a `Reduction` built with the same
positional/keyword arguments accepted by `Reduction`. An optional
`strategy` keyword overrides the default [`NeuralStrategy`](@ref).
"""
function NNHyperReduction(
  args...;
  model::NeuralNetwork=MultiLayerPerceptron(),
  strategy::NeuralStrategy=NeuralStrategy(model),
  kwargs...
  )

  reduction = Reduction(args...;kwargs...)
  NNHyperReduction(reduction,strategy)
end

RBSteady.get_reduction(r::NNHyperReduction) = r.reduction
get_strategy(r::NNHyperReduction) = r.strategy
