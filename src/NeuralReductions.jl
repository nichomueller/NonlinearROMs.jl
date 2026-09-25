"""
    struct NeuralReduction{A<:NeuralModel}
      model::A
      epochs::Int
      batch_size::Int
      sampler::MultiSampler
      optimiser::Optimiser
      trainlog::TrainingLog
    end

    NeuralReduction(
      model::NeuralModel;
      epochs::Int=20000,
      batch_size::Int=0,
      space_step=1,
      param_step=1,
      time_step=nothing,
      lr_scheduler=CosineAnnealing(epochs),
      verbose::Bool=true,
      print_every::Int=500,
      kwargs...
      )

The central configuration struct for training neural models. It defines the
neural architecture, the training hyperparameters, and the data subsampling
strategies for the offline phase. The keyword constructor is the intended
entry point: it builds the `sampler`/`optimiser`/`trainlog` fields from
`space_step`/`param_step`/`time_step`, `lr_scheduler`(`;kwargs...`), and
`verbose`/`print_every` respectively.

# Fields
- `model`: Any [`NeuralModel`](@ref) architecture — [`DeepONet`](@ref)/[`NOMAD`](@ref)
  (build one either with explicit `branch_layers`/`trunk_layers` (or
  `approximator_layers`/`decoder_layers`), or via the convenience
  `DeepONet(nbranch_in,ntrunk_in;width,depth,activation)` /
  `NOMAD(nsensors_in,ncoords_in;width,depth,activation)` constructors, which build a
  uniform stack of `depth` hidden layers of `width` neurons from the given input
  dimensions), [`MultiLayerPerceptron`](@ref), [`AutoEncoder`](@ref)/[`AutoDecoder`](@ref)/
  [`VariationalAutoEncoder`](@ref), or a [`KernelNeuralModel`](@ref). The
  [`DeepONetReduction`](@ref)/[`NOMADReduction`](@ref)/... aliases below pin `model`'s type
  to select which training pipeline `train`/`reduced_operator` dispatch to.
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

s = NeuralReduction(
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
strategy_log = NeuralReduction(
  DeepONet(2,3;width=64,depth=3), # 2 params -> Branch; 3D coords -> Trunk
  param_step = p -> log10.(p)
  )
```
"""
struct NeuralReduction{A<:NeuralModel}
  model::A
  epochs::Int
  batch_size::Int
  sampler::MultiSampler
  optimiser::Optimiser
  trainlog::TrainingLog
end

function NeuralReduction(
  model::NeuralModel;
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
  NeuralReduction(
    model,
    epochs,
    batch_size,
    sampler,
    optimiser,
    trainlog
  )
end

get_sampler(s::NeuralReduction) = s.sampler
get_optimiser(s::NeuralReduction) = s.optimiser.opt
get_scheduler(s::NeuralReduction) = s.optimiser.lr_scheduler
get_logger(s::NeuralReduction) = s.trainlog

function build_model(s::NeuralReduction)
  build_model(s.model)
end

function resolve_batch_size(s::NeuralReduction,(values,coords,params))
  bs = s.batch_size
  ns = size(params,2)
  bs > 0 ? min(bs,ns) : ns 
end

# interface for RB subspace + NN hyper-reduction machinery

struct NNRegressionStyle <: ReductionStyle end

"""
    struct NNRegression <: Reduction{NNRegressionStyle,EuclideanNorm}
      reduction::NeuralReduction
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
  reduction::NeuralReduction
  nparams::Int
end

function NNRegression(
  args...;
  nparams::Int=20,
  model::NeuralModel=MultiLayerPerceptron(),
  reduction::NeuralReduction=NeuralReduction(model),
  kwargs...
  )

  NNRegression(reduction,nparams)
end

ParamDataStructures.num_params(r::NNRegression) = r.nparams
RBSteady.get_reduction(r::NNRegression) = r.reduction

"""
    struct NNHyperReduction{A} <: HyperReduction{A}
      reduction::Reduction{A,EuclideanNorm}
      strategy::NeuralReduction
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
  strategy::NeuralReduction
end

"""
    NNHyperReduction(args...;model=MultiLayerPerceptron(),strategy=NeuralReduction(model),kwargs...) -> NNHyperReduction

Constructs a `NNHyperReduction` from a `Reduction` built with the same
positional/keyword arguments accepted by `Reduction`. An optional
`strategy` keyword overrides the default [`NeuralReduction`](@ref).
"""
function NNHyperReduction(
  args...;
  model::NeuralModel=MultiLayerPerceptron(),
  strategy::NeuralReduction=NeuralReduction(model),
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
  strategy::NeuralReduction
end

function TransientNNRegression(
  combination::TimeCombination,
  args...;
  nparams::Int=20,
  model::NeuralModel=MultiLayerPerceptron(),
  strategy::NeuralReduction=NeuralReduction(model),
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
  strategy::NeuralReduction
end

function TransientNNHyperReduction(
  combination::TimeCombination,
  args...;
  model::NeuralModel=MultiLayerPerceptron(),
  strategy::NeuralReduction=NeuralReduction(model),
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
    const KernelReduction{M<:KernelNeuralModel} = NeuralReduction{M}

A reduction wrapper for kernel-based neural models.
It instructs the ROM solvers to use the kernel neural model pipeline (tensor formatting and iterative integration) during the offline and online phases.
"""
const KernelReduction{M<:KernelNeuralModel} = NeuralReduction{M}

"""
    const DeepONetReduction{M<:DeepONet} = NeuralReduction{M}

A reduction wrapper for the Deep Operator Network (DeepONet).
It instructs the ROM solvers to use the DeepONet pipeline during the offline and online phases.

# Constructors
- `DeepONetReduction(s::NeuralReduction)`: Wraps an explicitly defined `NeuralReduction`.
- `DeepONetReduction(;model::DeepONet,kwargs...)`: Automatically builds the `NeuralReduction`, forwarding the training-hyperparameter keyword arguments to [`NeuralReduction`](@ref).

# Examples
```julia
# Using an explicit reduction
s = NeuralReduction(DeepONet(2,3;width=64,depth=3),epochs=1000)
reduction = DeepONetReduction(s)

# Using kwargs directly
reduction = DeepONetReduction(model=DeepONet(2,3;width=64,depth=3),epochs=1000,batch_size=32)
```
"""
const DeepONetReduction{M<:DeepONet} = NeuralReduction{M}

"""
    const NOMADReduction{M<:NOMAD} = NeuralReduction{M}

A reduction wrapper for the NOMAD (Non-linear Manifold Decoder) neural operator.
It instructs the ROM solvers to use the NOMAD pipeline during the offline and online phases.

# Constructors
- `NOMADReduction(s::NeuralReduction)`: Wraps an explicitly defined `NeuralReduction`.
- `NOMADReduction(;model::NOMAD,kwargs...)`: Automatically builds the `NeuralReduction`, forwarding the training-hyperparameter keyword arguments to [`NeuralReduction`](@ref).

# Examples
```julia
# Using an explicit reduction
s = NeuralReduction(NOMAD(2,3;width=32,depth=2),epochs=1000)
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
Same pattern as `DeepONetReduction`/`NOMADReduction`: wrap an explicit `NeuralReduction`,
or build one from `model=...`/kwargs directly, e.g.
`AutoEncoderReduction(model=AutoEncoder(width=32,depth=2),epochs=1000)`.
"""
const AutoEncoderReduction{M<:AutoEncoder} = NeuralReduction{M}
const AutoDecoderReduction{M<:AutoDecoder} = NeuralReduction{M}
const VAEReduction{M<:VariationalAutoEncoder} = NeuralReduction{M}

const MLPReduction{M<:MultiLayerPerceptron} = NeuralReduction{M}

for (f,m) in (
  (:KernelReduction,:KernelNeuralModel),
  (:DeepONetReduction,:DeepONet),
  (:NOMADReduction,:NOMAD),
  (:AutoEncoderReduction,:AutoEncoder),
  (:AutoDecoderReduction,:AutoDecoder),
  (:VAEReduction,:VariationalAutoEncoder),
  (:MLPReduction,:MultiLayerPerceptron),
)
  @eval begin
    $f(s::NeuralReduction{<:$m}) = s

    function $f(;model::$m,kwargs...)
      NeuralReduction(model;kwargs...)
    end
  end
end