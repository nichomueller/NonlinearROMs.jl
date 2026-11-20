abstract type NeuralSolver <: ROMSolver end

get_formatter(s::NeuralSolver) = @abstractmethod
get_optimiser(s::NeuralSolver) = @abstractmethod
get_scheduler(s::NeuralSolver) = @abstractmethod
get_batch_size(s::NeuralSolver) = @abstractmethod
get_loss(s::NeuralSolver) = @abstractmethod
get_logger(s::NeuralSolver) = @abstractmethod
build_model(s::NeuralSolver,args...;kwargs...) = @abstractmethod

function resolve_batch_size(s::NeuralSolver,(inputs,outputs))
  bs = get_batch_size(s)
  ns = num_samples(inputs)
  bs > 0 ? min(bs,ns) : ns
end

"""
    struct GenericNeuralSolver{A<:NeuralModel} <: NeuralSolver
      model::A
      epochs::Int
      batch_size::Int
      formatter::DataFormatter
      optimiser::Optimiser
      loss::Function
      trainlog::TrainingLog
    end

    NeuralSolver(
      model::NeuralModel;
      epochs::Int=20000,
      batch_size::Int=0,
      formatter=DataFormatter(model),
      lr_scheduler=CosineAnnealing(epochs),
      loss=Lux.MSELoss(),
      name=string(nameof(typeof(model))),
      verbose::Bool=true,
      print_every::Int=500,
      kwargs...
      )

The Reduced Basis Solver for Neural Operators: bundles a neural architecture
with its training hyperparameters. The keyword constructor is the intended
entry point: it builds the `optimiser`/`trainlog` fields from
`lr_scheduler`(`;kwargs...`) and `verbose`/`print_every` respectively.

# Arguments
- `model`: Any [`NeuralModel`](@ref) architecture — [`DeepONet`](@ref)/[`NOMAD`](@ref),
  [`MultiLayerPerceptron`](@ref), [`AutoEncoder`](@ref)/[`AutoDecoder`](@ref)/
  [`VariationalAutoEncoder`](@ref), or a [`KernelNeuralModel`](@ref).
- `epochs::Int`: Total number of training epochs. Default: `20000`.
- `batch_size::Int`: The batch size for training. If `0` (or negative), defaults
  to the total number of available samples (full-batch). Default: `0`.
- `formatter::DataFormatter`: Controls which of the model's `(coords,features)`
  inputs are passed/sampled. Default: `DataFormatter(model)`.
- `lr_scheduler`: The learning rate scheduler to use (e.g., `CosineAnnealing(epochs)`,
  `ReduceLROnPlateau()`). Default: `CosineAnnealing(epochs)`.
- `loss`: The training loss. Default: `Lux.MSELoss()`.
- `verbose::Bool`: If `true`, prints compilation times, training progress, and
  loss metrics. Default: `true`.
- `print_every::Int`: Frequency (in epochs) of the training progress output.
  Default: `500`.

# Examples

**Minimal Default Initialization:**
```julia
# Default hyperparameters (20000 epochs, full-batch, etc.)
solver = NeuralSolver(DeepONet(2,2))
```
**Custom Initialization:**
```julia
using Lux

solver = NeuralSolver(
  DeepONet(2,2;width=128,depth=4,activation=Lux.gelu),
  epochs=1000
)
```
"""
struct GenericNeuralSolver{A<:NeuralModel} <: NeuralSolver
  model::A
  epochs::Int
  batch_size::Int
  formatter::DataFormatter
  optimiser::Optimiser
  loss::Function
  trainlog::TrainingLog
end

function NeuralSolver(
  model::NeuralModel;
  epochs::Int=20000,
  batch_size::Int=0,
  formatter=DataFormatter(model),
  lr_scheduler=CosineAnnealing(epochs),
  loss=Lux.MSELoss(),
  name=string(nameof(typeof(model))),
  verbose::Bool=true,
  print_every::Int=500,
  kwargs...
  )

  optimiser = Optimiser(;lr_scheduler,kwargs...)
  trainlog = TrainingLog(name,epochs;verbose,print_every)
  GenericNeuralSolver(
    model,
    epochs,
    batch_size,
    formatter,
    optimiser,
    loss,
    trainlog
  )
end

get_formatter(s::GenericNeuralSolver) = s.formatter
get_optimiser(s::GenericNeuralSolver) = s.optimiser.opt
get_scheduler(s::GenericNeuralSolver) = s.optimiser.lr_scheduler
get_batch_size(s::GenericNeuralSolver) = s.batch_size
get_loss(s::GenericNeuralSolver) = s.loss
get_logger(s::GenericNeuralSolver) = s.trainlog
build_model(s::GenericNeuralSolver,args...;kwargs...) = build_model(s.model,args...;kwargs...)

struct RolloutNeuralSolver{A<:NeuralModel} <: NeuralSolver
  solver::GenericNeuralSolver{A}
  warmup_epochs::Int
  λ::Real
end

function RolloutNeuralSolver(args...;warmup_epochs::Int=0,λ::Real=1.0,kwargs...)
  solver = NeuralSolver(args...;kwargs...)
  RolloutNeuralSolver(solver,warmup_epochs,λ)
end

get_formatter(s::RolloutNeuralSolver) = get_formatter(s.solver)
get_optimiser(s::RolloutNeuralSolver) = get_optimiser(s.solver)
get_scheduler(s::RolloutNeuralSolver) = get_scheduler(s.solver)
get_batch_size(s::RolloutNeuralSolver) = get_batch_size(s.solver)
get_loss(s::RolloutNeuralSolver) = get_loss(s.solver)
get_logger(s::RolloutNeuralSolver) = get_logger(s.solver)
build_model(s::RolloutNeuralSolver,args...;kwargs...) = build_model(s.solver,args...;kwargs...)

"""
    struct NeuralOperator{A,B,C,D} <: AbstractLuxLayer
      chain::A
      parameters::B
      states::C
      stats::D
    end

A trained Lux `chain` bundled with its optimised parameters/states and the
normalisation `stats` (a [`Normalisation`](@ref)/[`ZScore`](@ref)) used to 
pre-normalise inputs and de-normalise outputs. Evaluable as `(a::NeuralOperator)(x) -> AbstractMatrix`.
"""
struct NeuralOperator{A,B,C,D} <: AbstractLuxLayer
  chain::A
  parameters::B
  states::C
  stats::D
end

function NeuralOperator(train_state::Lux.Training.TrainState,args...)
  parameters = train_state.parameters |> CDEV
  states = Lux.testmode(train_state.states) |> CDEV
  NeuralOperator(train_state.model,parameters,states,args...)
end

"""
    replace_space(op::NeuralOperator,space::DiscreteSpace) -> NeuralOperator

Returns `op` with every kernel's measure rebuilt on `space`, keeping the trained
parameters, states and normalisation: this is how a kernel operator trained on
some point cloud(s) is evaluated on a new one. `space` must be of the same kind as
the training one (graph-induced for a model trained on graphs, also when the
training space is a [`MultipleSpaces`](@ref) of graphs, e.g. a moving geometry), and
the inputs must then live on `get_coords(space)`.
"""
function replace_space(op::NeuralOperator,space::DiscreteSpace)
  chain = replace_measure(op.chain,InducedMeasure(space))
  NeuralOperator(chain,op.parameters,op.states,op.stats)
end

"""
    (m::NeuralOperator)(inputs) -> AbstractArray

Applies `m` to `inputs` (a `(params,coords)`/`(pin,xin)` tuple for DeepONet/NOMAD, or
a plain matrix): normalises `inputs` in place with `m.stats`, runs the Lux chain, then
rescales the prediction back with `m.stats`.
"""
function (m::NeuralOperator)(inputs)
  normalise!(inputs,m.stats)
  pred, = m.chain(inputs,m.parameters,m.states)
  denormalise!(pred,m.stats)
  return pred
end

"""
    reduced_operator(
      solver::NeuralSolver,
      feop::ParamFEOperator,
      s::AbstractSnapshots
    )

Executes the **Offline Phase** for Neural Operators on steady-state problems.
This method triggers the training loop of the neural network specified in the `solver`.

It automatically extracts the training dataset (parameters/sensors and spatial coordinates) from the snapshots `s` and the FE operator `feop`, normalizes the data, and performs the optimization.

Returns a `NeuralOperator` containing the trained network, its optimized weights, and the normalization statistics required for the online phase.
"""
function neural_operator(
  solver::NeuralSolver,
  input_space::AbstractInputSpace,
  output_space::AbstractOutputSpace
  )
  
  data,stats = fetch_data_and_stats(solver,input_space,output_space)
  train(solver,data,stats)
end

function neural_operator(
  solver::NeuralSolver,
  old_op::NeuralOperator,
  input_space::AbstractInputSpace,
  output_space::AbstractOutputSpace
  )
  
  data,stats = fetch_data_and_stats(solver,old_op,input_space,output_space)
  train(solver,data,stats)
end

function Algebra.solve(
  solver::NeuralSolver,
  op::NeuralOperator,
  input_space::AbstractInputSpace
  )

  data = fetch_data_from_stats(solver,input_space,op.stats)
  output, = op.chain(data,op.parameters,op.states)
  denormalise!(output,op.stats)
  return output
end

function Algebra.solve(
  solver::RolloutNeuralSolver,
  op::NeuralOperator,
  input_space::AbstractInputSpace,
  initial_output_space::AbstractOutputSpace
  )

  ic = get_values(initial_output_space)
  data = fetch_data_from_stats(solver,input_space,op.stats)
  output = rollout(op,data,ic)
  denormalise!(output,op.stats)
  return output
end

# rollout 

function rollout(op::NeuralOperator,data::TwoStageBuckets,ic::AbstractInputData)
  outputs = ()
  one_step_output = ic
  for _ in 1:num_clusters(data)
    pushforward_output = pushforward_step(op,data,one_step_output)
    one_step_output = one_step_step(op,data,pushforward_output)
    outputs = (outputs...,pushforward_output,one_step_output)
  end
  return outputs
end

function pushforward_step(op::NeuralOperator,data::TwoStageBuckets,one_step_output::AbstractInputData)
  pushforward_data = _merge_data(data.first_stage,one_step_output)
  pushforward_output, = op.chain(pushforward_data,op.parameters,op.states)
  denormalise!(pushforward_output,op.stats)
  return pushforward_output
end

function one_step_step(op::NeuralOperator,data::TwoStageBuckets,pushforward_output::AbstractInputData)
  one_step_data = _merge_data(data.second_stage,pushforward_output)
  one_step_output, = op.chain(one_step_data,op.parameters,op.states)
  denormalise!(one_step_output,op.stats)
  return one_step_output
end

# utils

fetch_data(solver::NeuralSolver,args...) = fetch_data(get_formatter(solver),args...)
fetch_stats(solver::NeuralSolver,args...) = fetch_stats(get_formatter(solver),args...)
merge_data(solver::NeuralSolver,args...) = merge_data(get_formatter(solver),args...)

function fetch_data_and_stats(
  solver::NeuralSolver,
  input_space::AbstractInputSpace,
  output_space::AbstractOutputSpace
  )

  raw_in = fetch_data(solver,input_space)
  input = (
    coords = isnothing(raw_in.coords) ? nothing : copy(raw_in.coords),
    features = isnothing(raw_in.features) ? nothing : copy(raw_in.features)
  )
  output = copy(fetch_data(solver,output_space))
  stats = fetch_stats(solver,(input,output))
  data = (input=merge_data(solver,input),output=output)
  return data,stats
end

function fetch_data_and_stats(
  solver::NeuralSolver,
  input_space::AbstractInputSpace
  )

  @notimplemented "Must provide output to properly build a normalisation factor"
end

function fetch_data_from_stats(
  solver::NeuralSolver,
  input_space::AbstractInputSpace,
  output_space::AbstractOutputSpace,
  stats::Normalisation
  )

  raw_in = fetch_data(solver,input_space)
  input = (
    coords = isnothing(raw_in.coords) ? nothing : copy(raw_in.coords),
    features = isnothing(raw_in.features) ? nothing : copy(raw_in.features)
  )
  output = copy(fetch_data(solver,output_space))
  normalise!(input,stats)
  normalise!(output,stats)
  return (input=merge_data(solver,input),output=output)
end

function fetch_data_from_stats(
  solver::NeuralSolver,
  input_space::AbstractInputSpace,
  stats::Normalisation
  )

  raw_in = fetch_data(solver,input_space)
  input = (
    coords = isnothing(raw_in.coords) ? nothing : copy(raw_in.coords),
    features = isnothing(raw_in.features) ? nothing : copy(raw_in.features)
  )
  normalise!(input,stats)
  return merge_data(solver,input)
end