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

using GridapROMs
using GridapROMs.DofMaps
using GridapROMs.ParamDataStructures
using GridapROMs.ParamODEs
using GridapROMs.ParamSteady
using GridapROMs.RBSteady
using GridapROMs.RBTransient
using GridapROMs.Utils

import GridapROMs.RBSteady:
  ROMSolver,GlobalRBSolver,GlobalContext,get_reduction,get_state_reduction,get_interpolation,
  allocate_coefficient,allocate_hyper_reduction,allocate_hypred_cache
import StaticArrays: SVector

export TrainingLog
export ZScore
export normalise!
export get_formatted_data
include("Utils.jl")

export LRScheduler
export CosineAnnealing
export ReduceLROnPlateau
export Optimiser
export step_scheduler!
export get_lr
include("LRSchedulers.jl")

export Sampler
export MultiSampler
export sample
export get_ids
export get_param_ids
export get_time_ids
include("Samplers.jl")

export MeshGraph
export DistanceGraph
export build_graph
export get_edge_tensors
include("GraphsInterface.jl")

export LatentCodeLayer
export VAELayer
export KernelNeuralLayer
include("NeuralLayers.jl")

export NeuralModel
export FiniteDimensionalModel
export CoordinateNeuralModel
export IntegralKernel
export DeepONet
export NOMAD
export MultiLayerPerceptron
export AutoEncoder
export VariationalAutoEncoder
export AutoDecoder
export build_model
export AbstractKernelModel
export KernelNeuralModel
export GNO
include("NeuralModels.jl")

export NNRegression
export NNHyperReduction
export TransientNNRegression
export TransientNNHyperReduction
export NeuralReduction
export DeepONetReduction
export NOMADReduction
export AutoEncoderReduction
export AutoDecoderReduction
export VAEReduction
export KernelReduction
export AbstractKernelReduction
export GNOReduction
include("NeuralReductions.jl")

export TrainedNeuralModel
export TrainedAutoEncoder
export TrainedAutoDecoder
export TrainedVAE
export train_model!
export infer_latent
export encode
export decode
export XDEV
export CDEV
include("NeuralModelsTraining.jl")

export NeuralSolver
export NeuralOperator
include("NeuralSolvers.jl")

export train
export resolve_batch_size
include("NeuralTraining.jl")

export NNRegressor
export NNContribution
export TransientNNRegressor
export TransientNNContribution
export TransientNNContributionTuple
include("HyperReductions.jl")

export NNInterpolation
include("Interpolations.jl")

include("RBOperators.jl")

end # module
