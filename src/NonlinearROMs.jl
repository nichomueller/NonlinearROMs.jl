"""
    module NonlinearROMs

Neural-network-based hyper-reduction and nonlinear reduced-order modelling
components for [`GridapROMs.jl`](https://github.com/gridap/GridapROMs.jl).

This package was extracted out of `GridapROMs.RBSteady`/`GridapROMs.RBTransient`
into its own repository, and plugs back into them via multiple dispatch:

- **Neural network models** (`NeuralModels.jl`) — `DeepONet`, `NOMAD`,
  `MultiLayerPerceptron`, `AutoEncoder`, `VariationalAutoEncoder`, `AutoDecoder`,
  `GenericNeuralModel`; all trained through the same Lux/Reactant/Enzyme
  pipeline (`NeuralReduction`, `train_model!`, `train_neural_coefficient`).

- **Steady hyper-reduction** — `NNOperatorReduction` (operator regression),
  `NNHyperReduction` (NN-predicted EIM coefficients), `NNOperator`,
  `NNInterpolation`, extending `RBSteady.HRProjection`/`RBSteady.Interpolation`
  and the `Algebra.residual!`/`Algebra.jacobian!` dispatch for `RBOperator`.

- **Transient hyper-reduction** — `TransientNNOperatorReduction`,
  `TransientNNHyperReduction`, the transient counterparts extending
  `RBTransient`'s space-time hyper-reduction machinery analogously.

Usage: construct a `NNHyperReduction`/`NNOperatorReduction`
(or their `Transient*` transient counterparts) and pass it to `RBSolver` wherever
a steady/transient `HyperReduction` is expected, exactly as you would
`MDEIMHyperReduction` or `RBFHyperReduction`.
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
  GlobalRBSolver,GlobalContext,get_reduction,get_state_reduction,get_interpolation,
  allocate_coefficient,allocate_hyper_reduction,allocate_hypred_cache
import StaticArrays: SVector

export TrainingLog
export ZscoreStats
export normalise!
export CoordinateSnapshots
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
include("GraphsInterface.jl")

export LatentCodeLayer
export VAELayer
export NeuralOperatorLayer
include("NeuralLayers.jl")

export NeuralModel
export FiniteDimensionalModel
export AbstractNeuralOperator
export AbstractCoordinateBasedOperator
export AbstractKernelNeuralOperator
export AbstractIntegralKernel
export GenericNeuralModel
include("NeuralModels.jl")

export DeepONet
export NOMAD
export MultiLayerPerceptron
export AutoEncoder
export VariationalAutoEncoder
export AutoDecoder
export build_model
export KernelNeuralOperator
include("NeuralModels.jl")

export NNOperatorReduction
export NNHyperReduction
export TransientNNOperatorReduction
export TransientNNHyperReduction
export NeuralReduction
export DeepONetReduction
export NOMADReduction
export AutoEncoderReduction
export AutoDecoderReduction
export VAEReduction
export KernelOperatorReduction
include("NeuralReductions.jl")

export TrainedNeuralModel
export TrainedAutoEncoder
export TrainedAutoDecoder
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
export train_deeponet!
export train_nomad!
export train_autoencoder!
export train_autodecoder!
export train_vae!
export train_neural_coefficient
export TrainedVAE
export resolve_batch_size
include("NeuralTraining.jl")

export NNHRProjection
export NNOperator
export NNContribution
export TransientNNProjection
export TransientNNContribution
export TransientNNContributionTuple
include("HyperReductions.jl")

export NNInterpolation
include("Interpolations.jl")

include("RBOperators.jl")

end # module
