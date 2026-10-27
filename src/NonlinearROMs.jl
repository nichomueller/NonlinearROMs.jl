"""
    module NonlinearROMs

Neural-network-based surrogate modelling and hyper-reduction components for
[`GridapROMs.jl`](https://github.com/gridap/GridapROMs.jl).

This package was extracted out of `GridapROMs.RBSteady`/`GridapROMs.RBTransient`
into its own repository, and plugs back into them via multiple dispatch:

- **Neural network models** (`NeuralModels.jl`/`NeuralLayers.jl`) — `DeepONet`,
  `NOMAD`, `MultiLayerPerceptron`, `AutoEncoder`, `VariationalAutoEncoder`,
  `AutoDecoder`, `KernelNeuralModel`; all trained through the same
  Lux/Reactant/Enzyme pipeline (`NeuralReduction`, `train_model!`).

- **Direct neural surrogates** — [`NeuralSolver`](@ref)/[`NeuralOperator`](@ref)
  train a [`NeuralReduction`](@ref)'s model (through its `DeepONetReduction`/
  `NOMADReduction`/`KernelReduction`/`AutoEncoderReduction`/`AutoDecoderReduction`/
  `VAEReduction`/`MLPReduction` alias) to map parameters directly to the
  full-order solution, bypassing FE assembly entirely; `reduced_operator`
  plugs this into the same offline/online API as `GridapROMs`' RB solvers.

- **Steady hyper-reduction** — `NNRegression` (operator regression),
  `NNHyperReduction` (NN-predicted EIM coefficients), `NNRegressor`,
  `NNInterpolation`, extending `RBSteady.HRProjection`/`RBSteady.Interpolation`
  and the `Algebra.residual!`/`Algebra.jacobian!` dispatch for `RBOperator`.

- **Transient hyper-reduction** — `TransientNNRegression`,
  `TransientNNHyperReduction`, the transient counterparts extending
  `RBTransient`'s space-time hyper-reduction machinery analogously.

- **Mesh graph utilities** (`GraphsInterface.jl`/`WeightedSimpleDiGraphs.jl`) —
  `build_graph(coords;radius)` turns a point set into a radius-neighbourhood
  `WeightedSimpleDiGraph` (edge weights = Euclidean distance), wrapped in a
  [`GraphInducedSpace`](@ref), for graph-based neural architectures.

Usage: construct a `NNHyperReduction`/`NNRegression` (or their `Transient*`
counterparts) and pass it to `RBSolver` wherever a steady/transient
`HyperReduction` is expected, exactly as you would `MDEIMHyperReduction` or
`RBFHyperReduction`; or construct a `NeuralSolver` directly and call
`reduced_operator`/`solve` on it as you would any other ROM solver.
"""
module NonlinearROMs

using BlockArrays
using Enzyme
using FFTW
using FillArrays
using ForwardDiff
using Graphs
using LinearAlgebra
using Lux
using MLUtils
using NNlib
using NearestNeighbors
using Optimisers
using Random
using Reactant
using SparseArrays
using Statistics

using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.CellData
using Gridap.FESpaces
using Gridap.Geometry
using Gridap.Helpers
using Gridap.MultiField
using Gridap.Polynomials
using Gridap.ReferenceFEs
using Gridap.TensorValues

using GridapROMs
using GridapROMs.DofMaps
using GridapROMs.ParamDataStructures
using GridapROMs.ParamFESpaces
using GridapROMs.ParamODEs
using GridapROMs.ParamSteady
using GridapROMs.RBSteady
using GridapROMs.RBTransient
using GridapROMs.Utils

import Lux: AbstractLuxLayer
import Gridap.Fields: GenericField
import GridapROMs.RBSteady: get_interpolation,allocate_coefficient,allocate_hyper_reduction,allocate_hypred_cache
import Random: AbstractRNG
import StaticArrays: MVector,SVector

export TrainingLog
export ZScore
export EnergyLoss
export visualise
include("Utils.jl")

export LRScheduler
export CosineAnnealing
export ReduceLROnPlateau
export Optimiser
include("LRSchedulers.jl")

export FeatureBundle
export DataFormatter
export Sampler
include("DataInterface.jl")

export WeightedSimpleDiGraph
export NystromSubgraph
export gather_data
export scatter_data
export attention
export build_graph
include("GraphsInterface.jl")

export DiscreteSpace
export TrainingSpace
export InputSpace
export OutputSpace
export FunctionSpace
export GraphInducedSpace
export MultiGraphInducedSpace
export get_coords
export get_features
export get_graph
export to_element
include("DiscreteSpaces.jl")

export InducedMeasure
export FESpaceMeasure
export GraphMeasure
export get_space
export integration
include("InducedMeasures.jl")

export AbstractIntegralKernel
export IntegralKernel
export FourierKernel
export AttentionKernel
export MultipleAttentionKernel
export KernelNeuralLayer
export Lift
export Project
include("IntegralKernels.jl")

export RoPELayer
include("NeuralLayers.jl")

export NeuralModel
export KernelStyle
export IntegralStyle
export FourierStyle
export KernelNeuralModel
export NOMAD
export MultiLayerPerceptron
include("NeuralModels.jl")

export NNRegression
export NNHyperReduction
export TransientNNRegression
export TransientNNHyperReduction
export NeuralReduction
export NOMADReduction
export KernelReduction
include("NeuralReductions.jl")

export NeuralSolver
export NeuralOperator
export neural_operator
include("NeuralOperators.jl")

export collect_trajectory
export rollout_data
export rollout
include("Rollout.jl")

export train
export resolve_batch_size
include("NeuralTraining.jl")

end # module
