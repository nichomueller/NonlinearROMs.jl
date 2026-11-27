"""
    module NonlinearROMs

Neural-operator surrogates for parameterised PDEs, built on
[`Gridap.jl`](https://github.com/gridap/Gridap.jl) and
[`GridapROMs.jl`](https://github.com/gridap/GridapROMs.jl).

- **Data** — [`InputSpace`](@ref)/[`OutputSpace`](@ref) evaluate coordinates, a
  [`FeatureBundle`](@ref) of parameterised coefficients and FE solutions as
  `(D,N,K)` arrays; [`to_element`](@ref) maps predictions back to `FEFunction`s.

- **Models** — [`KernelNeuralModel`](@ref): kernel neural operators, with the
  integral kernel discretised by FE quadrature ([`FunctionSpace`](@ref)) or on a
  radius graph ([`GraphInducedSpace`](@ref), [`MultiGraphInducedSpace`](@ref)),
  or spectral ([`FourierStyle`](@ref)); multi-field problems and several meshes
  are handled by gluing local spaces into a [`MultipleSpaces`](@ref).

- **Training** — [`NeuralSolver`](@ref)/[`neural_operator`](@ref) train through
  a Lux/Reactant/Enzyme pipeline, with configurable normalisation and
  [`EnergyLoss`](@ref)es in the norms of the FE space;
  [`RolloutNeuralSolver`](@ref) trains autoregressive operators for transient
  problems ([`RolloutData`](@ref)).
"""
module NonlinearROMs

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

import DataStructures: OrderedDict
import Gridap.Fields: GenericField
import GridapROMs.ParamODEs: TransientParamFESolution
import GridapROMs.RBSteady: get_interpolation,allocate_coefficient,allocate_hyper_reduction,allocate_hypred_cache
import Lux: AbstractLuxLayer
import Random: AbstractRNG
import SparseArrays: AbstractSparseMatrix
import StaticArrays: MVector,SVector

export TrainingLog
export ZScore
export MScoreStyle
export RScoreStyle
export ZScoreStyle
export EnergyLoss
export visualise
export normalise!
export denormalise!
include("Utils.jl")

export LRScheduler
export CosineAnnealing
export ReduceLROnPlateau
export Optimiser
include("LRSchedulers.jl")

export FeatureBundle
export DataFormatter
export SpatialKernelFormatter
export FeatureKernelFormatter
export StateKernelFormatter
export FullKernelFormatter
export Sampler
export cat_channels
export cat_samples
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
export MultipleSpaces
export get_coords
export get_features
export get_graph
export get_message_rule
export to_element
include("DiscreteSpaces.jl")

export InducedMeasure
export FESpaceMeasure
export GraphMeasure
export MPRule
export PlainMP
export AnchoredMP
export DifferenceMP
export CompositeMP
export combine_message
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

export NeuralSolver
export RolloutNeuralSolver
export NeuralOperator
export neural_operator
export replace_space
include("NeuralOperators.jl")

export RolloutData
export rollout_data
export rollout
include("Rollout.jl")

export train
export resolve_batch_size
include("NeuralTraining.jl")

end # module
