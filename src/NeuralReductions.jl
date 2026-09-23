# interface for RB subspace + NN hyper-reduction machinery 

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

# transient 

abstract type AbstractTransientNNHyperReduction{A} <: TransientHyperReduction{A} end

"""
    struct TransientNNOperatorReduction <: AbstractTransientNNHyperReduction{NoReductionStyle}

Transient counterpart of [`NNOperatorReduction`](@ref). The NN maps parameter
values to the time-combined Galerkin-projected Jacobian, bypassing FE assembly.
Carry the `combination::TimeCombination` from the ODE solver.
"""
struct TransientNNOperatorReduction <: AbstractTransientNNHyperReduction{NoReductionStyle}
  combination::TimeCombination
  nparams::Int
  strategy::NeuralStrategy
end

function TransientNNOperatorReduction(
  combination::TimeCombination,
  args...;
  nparams::Int=20,
  model::NeuralNetwork=MultiLayerPerceptron(),
  strategy::NeuralStrategy=NeuralStrategy(model),
  kwargs...
  )

  TransientNNOperatorReduction(combination,nparams,strategy)
end

ParamDataStructures.num_params(r::TransientNNOperatorReduction) = r.nparams
get_strategy(r::TransientNNOperatorReduction) = r.strategy
RBTransient.get_time_combination(r::TransientNNOperatorReduction) = r.combination

"""
    struct TransientNNHyperReduction{A} <: AbstractTransientNNHyperReduction{A}

Transient counterpart of [`NNHyperReduction`](@ref). The NN predicts EIM
coefficients from parameter values; the time combination is applied at the
basis-projection stage.
"""
struct TransientNNHyperReduction{A} <: AbstractTransientNNHyperReduction{A}
  combination::TimeCombination
  reduction::Reduction{A,EuclideanNorm}
  strategy::NeuralStrategy
end

function TransientNNHyperReduction(
  combination::TimeCombination,
  args...;
  model::NeuralNetwork=MultiLayerPerceptron(),
  strategy::NeuralStrategy=NeuralStrategy(model),
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
    Base.@kwdef struct NeuralStrategy{M,S}
      model::M
      epochs::Int = 20000
      batch_size::Int = 0
      space_step = 1
      param_step = 1
      time_step = nothing
      lr_scheduler::S = CosineAnnealing(epochs)
      verbose::Bool = true
      print_every::Int = 500
    end

The central configuration struct for training Neural Operators. It defines the
neural architecture, the training hyperparameters, and the data subsampling
strategies for the offline phase.

# Fields
- `model`: A [`DeepONet`](@ref) or [`NOMAD`](@ref) architecture. Build one either with
  explicit `branch_layers`/`trunk_layers` (or `approximator_layers`/`decoder_layers`), or
  via the convenience `DeepONet(nbranch_in,ntrunk_in;width,depth,activation)` /
  `NOMAD(nsensors_in,ncoords_in;width,depth,activation)` constructors, which build a
  uniform stack of `depth` hidden layers of `width` neurons from the given input dimensions.
- `epochs::Int`: Total number of training epochs. Default: `20000`.
- `batch_size::Int`: The batch size for training. If set to `0` or a negative value, it defaults to the total number of available samples (full-batch). Default: `0`.
- `sampler::MultiSampler`: Built from `space_step`/`param_step`/`time_step`, controls how the
  spatial DoFs, the parameters, and (for transient problems) the time steps are subsampled
  from the full-order data before feeding it to the network. Each of `space_step`/`param_step`/
  `time_step` can be an `Integer` (stride), a `Function` (a per-sample transform, e.g.
  `p -> log10.(p)`), an `AbstractVector` of explicit indices, or `nothing`/`identity` (no
  subsampling). Default: `space_step=1`, `param_step=1`, `time_step=nothing`.
- `lr_scheduler`: The learning rate scheduler to use (e.g., `CosineAnnealing(epochs)`, `ReduceLROnPlateau()`). Default: `CosineAnnealing(epochs)`.
- `verbose::Bool`: If `true`, prints compilation times, training progress, and loss metrics. Default: `true`.
- `print_every::Int`: Frequency (in epochs) of the training progress output. Default: `500`.

# Examples

**Basic Usage:**
```julia
using Lux

s = NeuralStrategy(
  DeepONet(2,2;width=64,depth=3,activation=Lux.gelu), # 2 params -> Branch; 2D coords -> Trunk
  epochs = 5000,
  batch_size = 32,
  space_step = 2, # Use half of the spatial DoFs for training
  lr_scheduler = CosineAnnealing(5000,lr_max=1e-3,lr_min=1e-6)
  )
```

**Advanced Usage (Multi-Scale Learning):**
```julia
# Log-transform for parameters spanning huge ranges (e.g., 1e-(beta) with beta = 1:0.2:5)
strategy_log = NeuralStrategy(
  DeepONet(2,3;width=64,depth=3), # 2 params -> Branch; 3D coords -> Trunk
  param_step = p -> log10.(p)
  )
```
"""
struct NeuralStrategy{A<:NeuralNetwork}
  model::A
  epochs::Int
  batch_size::Int
  sampler::MultiSampler
  optimiser::Optimiser
  trainlog::TrainingLog
end

function NeuralStrategy(
  model::NeuralNetwork;
  epochs::Int=20000,
  batch_size::Int=0,
  space_step=1,
  param_step=1,
  time_step=nothing,
  lr_scheduler=CosineAnnealing(epochs),
  name=string(nameof(typeof(model))),
  verbose::Bool=true,
  print_every::Int=500,
  kwargs...
  )

  sampler = MultiSampler(;space_step,param_step,time_step)
  optimiser = Optimiser(;lr_scheduler,kwargs...)
  trainlog = TrainingLog(name,epochs;verbose,print_every)
  NeuralStrategy(
    model,
    epochs,
    batch_size,
    sampler,
    optimiser,
    trainlog
  )
end

get_sampler(s::NeuralStrategy) = s.sampler
get_optimiser(s::NeuralStrategy) = s.optimiser.opt
get_scheduler(s::NeuralStrategy) = s.optimiser.lr_scheduler
get_logger(s::NeuralStrategy) = s.trainlog

function build_model(s::NeuralStrategy)
  build_model(s.model)
end

struct NeuralReduction{A<:NeuralNetwork} <: Reduction{NoReductionStyle,EuclideanNorm}
  s::NeuralStrategy{A}
end

RBSteady.ReductionStyle(r::NeuralReduction) = NoReductionStyle()
RBSteady.NormStyle(r::NeuralReduction) = EuclideanNorm()
get_strategy(r::NeuralReduction) = r.s

"""
    const KernelOperatorReduction{M<:AbstractKernelNeuralOperator} = NeuralReduction{M}

A reduction wrapper for Kernel-based Neural Operators.
It instructs the ROM solvers to use the Kernel Neural Operator pipeline (tensor formatting and iterative integration) during the offline and online phases.
"""
const KernelOperatorReduction{M<:AbstractKernelNeuralOperator} = NeuralReduction{M}

"""
    const DeepONetReduction{M<:DeepONet} = NeuralReduction{M}

A reduction wrapper for the Deep Operator Network (DeepONet) s.
It instructs the ROM solvers to use the DeepONet pipeline during the offline and online phases.

# Constructors
- `DeepONetReduction(s::NeuralStrategy)`: Wraps an explicitly defined s.
- `DeepONetReduction(;model::DeepONet,kwargs...)`: Automatically builds the s, forwarding the training-hyperparameter keyword arguments to [`NeuralStrategy`](@ref).

# Examples
```julia
# Using an explicit s
s = NeuralStrategy(DeepONet(2,3;width=64,depth=3),epochs=1000)
reduction = DeepONetReduction(s)

# Using kwargs directly
reduction = DeepONetReduction(model=DeepONet(2,3;width=64,depth=3),epochs=1000,batch_size=32)
```
"""
const DeepONetReduction{M<:DeepONet} = NeuralReduction{M}

"""
    const NOMADReduction{M<:NOMAD} = NeuralReduction{M}

A reduction wrapper for the NOMAD (Non-linear Manifold Decoder) neural operator s.
It instructs the ROM solvers to use the NOMAD pipeline during the offline and online phases.

# Constructors
- `NOMADReduction(s::NeuralStrategy)`: Wraps an explicitly defined s.
- `NOMADReduction(;model::NOMAD,kwargs...)`: Automatically builds the s, forwarding the training-hyperparameter keyword arguments to [`NeuralStrategy`](@ref).

# Examples
```julia
# Using an explicit s
s = NeuralStrategy(NOMAD(2,3;width=32,depth=2),epochs=1000)
reduction = NOMADReduction(s)

# Using kwargs directly
reduction = NOMADReduction(model=NOMAD(2,3;width=32,depth=2),epochs=1000)
```
"""
const NOMADReduction{M<:NOMAD} = NeuralReduction{M}

"""
    const AutoEncoderReduction{M<:AutoEncoder} = NeuralReduction{M}
    const AutoDecoderReduction{M<:AutoDecoder} = NeuralReduction{M}
    const VAEReduction{M<:VariationalAutoEncoder} = NeuralReduction{M}

Reduction wrappers for the reconstruction-based architectures, analogous to
[`DeepONetReduction`](@ref)/[`NOMADReduction`](@ref): they instruct
[`reduced_operator`](@ref) to train an [`AutoEncoder`](@ref)/[`AutoDecoder`](@ref)/
[`VariationalAutoEncoder`](@ref) on the snapshot data itself (no parameters/coordinates
involved), producing a [`NeuralOperator`](@ref) with `metadata === identity`.

# Constructors
Same pattern as `DeepONetReduction`/`NOMADReduction`: wrap an explicit `NeuralStrategy`,
or build one from `model=...`/kwargs directly, e.g.
`AutoEncoderReduction(model=AutoEncoder(width=32,depth=2),epochs=1000)`.
"""
const AutoEncoderReduction{M<:AutoEncoder} = NeuralReduction{M}
const AutoDecoderReduction{M<:AutoDecoder} = NeuralReduction{M}
const VAEReduction{M<:VariationalAutoEncoder} = NeuralReduction{M}

for (f,m) in (
  (:DeepONetReduction,:DeepONet),
  (:NOMADReduction,:NOMAD),
  (:AutoEncoderReduction,:AutoEncoder),
  (:AutoDecoderReduction,:AutoDecoder),
  (:VAEReduction,:VariationalAutoEncoder),
  (:KernelOperatorReduction,:AbstractKernelNeuralOperator)
)
  @eval begin
    $f(s::NeuralStrategy{<:$m}) = NeuralReduction(s)

    function $f(;model::$m,kwargs...)
      s = NeuralStrategy(model;kwargs...)
      NeuralReduction(s)
    end
  end
end