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
  `MeshGraph`/`DistanceGraph`/`build_graph` turn a `FESpace`'s dof adjacency (or
  a coordinate-based nearest-neighbour graph) into a `WeightedSimpleDiGraph`,
  for graph-based neural architectures.

Usage: construct a `NNHyperReduction`/`NNRegression` (or their `Transient*`
counterparts) and pass it to `RBSolver` wherever a steady/transient
`HyperReduction` is expected, exactly as you would `MDEIMHyperReduction` or
`RBFHyperReduction`; or construct a `NeuralSolver` directly and call
`reduced_operator`/`solve` on it as you would any other ROM solver.
"""
module NonlinearROMs

using Enzyme
using FillArrays
using ForwardDiff
using Graphs
using LinearAlgebra
using Lux
using MLUtils
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
import GridapROMs.RBSteady:
  GlobalRBSolver,GlobalContext,get_reduction,get_state_reduction,get_interpolation,
  allocate_coefficient,allocate_hyper_reduction,allocate_hypred_cache
import Random: AbstractRNG
import StaticArrays: MVector,SVector

export TrainingLog
export ZScore
include("Utils.jl")

export LRScheduler
export CosineAnnealing
export ReduceLROnPlateau
export Optimiser
include("LRSchedulers.jl")

export DiscreteSpace
export FunctionSpace
export GraphInducedSpace
export SimpleSpace
export MultiSpace
include("DiscreteSpaces.jl")

export WeightedSimpleDiGraph
export CoordinateGraph
export NystromSubspace
export gather_data
export scatter_data
export attention
include("GraphsInterface.jl")

export FeatureBundle
export DataFormatter
include("DataInterface.jl")

export NeuralModel
export KernelNeuralModel
export NOMAD
export MultiLayerPerceptron
export TrainedNeuralModel
include("NeuralModels.jl")

export NNRegression
export NNHyperReduction
export TransientNNRegression
export TransientNNHyperReduction
export NeuralReduction
export NOMADReductionon
export KernelReduction
include("NeuralReductions.jl")

export NeuralSolver
export NeuralOperator
include("NeuralSolvers.jl")

export train
export resolve_batch_size
include("NeuralTraining.jl")

end # module
