# interface for RB subspace + NN hyper-reduction machinery

struct NNRegressionStyle <: ReductionStyle end

"""
    struct NNRegression <: Reduction{NNRegressionStyle,EuclideanNorm}
      reduction::NeuralSolver
      nparams::Int
    end

A hyper-reduction strategy for **operator regression**: the NN directly maps
parameter values to the Galerkin-projected residual vector, bypassing FE
assembly entirely during the online phase. Only suitable for residual
(vector-valued) operators.

The offline phase projects the residual snapshots onto the test space and
trains the NN to reproduce the projected vectors. The online phase calls
the NN forward pass, producing the projected residual without any assembly.

`reduction` controls the [`MultiLayerPerceptron`](@ref) architecture and training.
`nparams` controls how many parameter samples to use for NN training.
"""
struct NNRegression <: Reduction{NNRegressionStyle,EuclideanNorm}
  reduction::NeuralSolver
  nparams::Int
end

function NNRegression(
  args...;
  nparams::Int=20,
  model::NeuralModel=MultiLayerPerceptron(),
  reduction::NeuralSolver=NeuralSolver(model),
  kwargs...
  )

  NNRegression(reduction,nparams)
end

ParamDataStructures.num_params(r::NNRegression) = r.nparams
RBSteady.get_reduction(r::NNRegression) = r.reduction

"""
    struct NNHyperReduction{A} <: HyperReduction{A}
      reduction::Reduction{A,EuclideanNorm}
      strategy::NeuralSolver
    end

A hyper-reduction strategy that uses a neural network to predict EIM
coefficients from parameter values. The offline phase:

1. applies empirical interpolation on the snapshot basis to extract coefficients
2. trains a [`MultiLayerPerceptron`](@ref) via `strategy` on the `(μ,coefficient)` pairs

The online phase calls the NN forward pass instead of assembling the FE
operator on the reduced integration domain.
"""
struct NNHyperReduction{A} <: HyperReduction{A}
  reduction::Reduction{A,EuclideanNorm}
  strategy::NeuralSolver
end

"""
    NNHyperReduction(args...;model=MultiLayerPerceptron(),strategy=NeuralSolver(model),kwargs...) -> NNHyperReduction

Constructs a `NNHyperReduction` from a `Reduction` built with the same
positional/keyword arguments accepted by `Reduction`. An optional
`strategy` keyword overrides the default [`NeuralSolver`](@ref).
"""
function NNHyperReduction(
  args...;
  model::NeuralModel=MultiLayerPerceptron(),
  strategy::NeuralSolver=NeuralSolver(model),
  kwargs...
  )

  reduction = Reduction(args...;kwargs...)
  NNHyperReduction(reduction,strategy)
end

RBSteady.get_reduction(r::NNHyperReduction) = r.reduction
get_strategy(r::NNHyperReduction) = r.strategy

# transient 

abstract type AbstractTransientNNHyperReduction{A} <: TransientHyperReduction{A} end

"""
    struct TransientNNRegression <: AbstractTransientNNHyperReduction{NoReductionStyle}

Transient counterpart of [`NNRegression`](@ref). The NN maps parameter
values to the time-combined Galerkin-projected Jacobian, bypassing FE assembly.
Carry the `combination::TimeCombination` from the ODE solver.
"""
struct TransientNNRegression <: AbstractTransientNNHyperReduction{NoReductionStyle}
  combination::TimeCombination
  nparams::Int
  strategy::NeuralSolver
end

function TransientNNRegression(
  combination::TimeCombination,
  args...;
  nparams::Int=20,
  model::NeuralModel=MultiLayerPerceptron(),
  strategy::NeuralSolver=NeuralSolver(model),
  kwargs...
  )

  TransientNNRegression(combination,nparams,strategy)
end

ParamDataStructures.num_params(r::TransientNNRegression) = r.nparams
get_strategy(r::TransientNNRegression) = r.strategy
RBTransient.get_time_combination(r::TransientNNRegression) = r.combination

"""
    struct TransientNNHyperReduction{A} <: AbstractTransientNNHyperReduction{A}

Transient counterpart of [`NNHyperReduction`](@ref). The NN predicts EIM
coefficients from parameter values; the time combination is applied at the
basis-projection stage.
"""
struct TransientNNHyperReduction{A} <: AbstractTransientNNHyperReduction{A}
  combination::TimeCombination
  reduction::Reduction{A,EuclideanNorm}
  strategy::NeuralSolver
end

function TransientNNHyperReduction(
  combination::TimeCombination,
  args...;
  model::NeuralModel=MultiLayerPerceptron(),
  strategy::NeuralSolver=NeuralSolver(model),
  kwargs...
  )

  reduction = Reduction(args...;kwargs...)
  TransientNNHyperReduction(combination,reduction,strategy)
end

RBSteady.get_reduction(r::TransientNNHyperReduction) = r.reduction
get_strategy(r::TransientNNHyperReduction) = r.strategy
RBTransient.get_time_combination(r::TransientNNHyperReduction) = r.combination

# neural reductions

"""
    const KernelReduction{M<:KernelNeuralModel} = NeuralSolver{M}

A reduction wrapper for kernel-based neural models.
It instructs the ROM solvers to use the kernel neural model pipeline (tensor formatting and iterative integration) during the offline and online phases.
"""
const KernelReduction{M<:KernelNeuralModel} = NeuralSolver{M}

"""
    const NOMADReduction{M<:NOMAD} = NeuralSolver{M}

A reduction wrapper for the NOMAD (Non-linear Manifold Decoder) neural operator.
It instructs the ROM solvers to use the NOMAD pipeline during the offline and online phases.

# Constructors
- `NOMADReduction(s::NeuralSolver)`: Wraps an explicitly defined `NeuralSolver`.
- `NOMADReduction(;model::NOMAD,kwargs...)`: Automatically builds the `NeuralSolver`, forwarding the training-hyperparameter keyword arguments to [`NeuralSolver`](@ref).

# Examples
```julia
# Using an explicit solver
s = NeuralSolver(NOMAD(2,3;width=32,depth=2),epochs=1000)
reduction = NOMADReduction(s)

# Using kwargs directly
reduction = NOMADReduction(model=NOMAD(2,3;width=32,depth=2),epochs=1000)
```
"""
const NOMADReduction{M<:NOMAD} = NeuralSolver{M}

const MLPReduction{M<:MultiLayerPerceptron} = NeuralSolver{M}

for (f,m) in (
  (:KernelReduction,:KernelNeuralModel),
  (:NOMADReduction,:NOMAD),
  (:MLPReduction,:MultiLayerPerceptron),
)
  @eval begin
    $f(s::NeuralSolver{<:$m}) = s

    function $f(;model::$m,kwargs...)
      NeuralSolver(model;kwargs...)
    end
  end
end