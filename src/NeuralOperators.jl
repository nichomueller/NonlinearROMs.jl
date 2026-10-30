"""
    struct NeuralSolver{A<:NeuralModel} <: ROMSolver
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
struct NeuralSolver{A<:NeuralModel} <: ROMSolver
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
  NeuralSolver(
    model,
    epochs,
    batch_size,
    formatter,
    optimiser,
    loss,
    trainlog
  )
end

get_formatter(s::NeuralSolver) = s.formatter
get_optimiser(s::NeuralSolver) = s.optimiser.opt
get_scheduler(s::NeuralSolver) = s.optimiser.lr_scheduler
get_loss(s::NeuralSolver) = s.loss
get_logger(s::NeuralSolver) = s.trainlog
build_model(s::NeuralSolver,args...;kwargs...) = build_model(s.model,args...;kwargs...)

function resolve_batch_size(s::NeuralSolver,(inputs,outputs))
  bs = s.batch_size
  ns = num_samples(inputs)
  bs > 0 ? min(bs,ns) : ns
end

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
  input_space::InputSpace,
  output_space::OutputSpace
  )
  
  data,stats = fetch_data_and_stats(solver,input_space,output_space)
  train(solver,data,stats)
end

function neural_operator(
  solver::NeuralSolver,
  old_op::NeuralOperator,
  input_space::InputSpace,
  output_space::OutputSpace
  )
  
  data,stats = fetch_data_and_stats(solver,old_op,input_space,output_space)
  train(solver,data,stats)
end

function Algebra.solve(
  solver::NeuralSolver,
  op::NeuralOperator,
  input_space::InputSpace
  )

  data = fetch_data_from_stats(solver,input_space,op.stats)
  output, = op.chain(data,op.parameters,op.states)
  denormalise!(output,op.stats)
  return output
end

function Algebra.solve(
  solver::NeuralSolver,
  op::NeuralOperator,
  input_space::InputSpace,
  output_space::OutputSpace
  )

  output = solve(solver,op,input_space)
  to_element(output_space,output)
end

# utils

fetch_data(solver::NeuralSolver,args...) = fetch_data(get_formatter(solver),args...)
fetch_stats(solver::NeuralSolver,args...) = fetch_stats(get_formatter(solver),args...)
merge_data(solver::NeuralSolver,args...) = merge_data(get_formatter(solver),args...)

function fetch_data_and_stats(
  solver::NeuralSolver,
  input_space::InputSpace,
  output_space::OutputSpace
  )

  input = fetch_data(solver,input_space)
  output = fetch_data(solver,output_space)
  stats = fetch_stats(solver,(input,output))
  data = (input=merge_data(solver,input),output=output)
  return data,stats
end

function fetch_data_and_stats(
  solver::NeuralSolver,
  input_space::InputSpace
  )

  @notimplemented "Must provide output to properly build a normalisation factor"
end

function fetch_data_from_stats(
  solver::NeuralSolver,
  input_space::InputSpace,
  output_space::OutputSpace,
  stats::Normalisation
  )

  input = fetch_data(solver,input_space)
  output = fetch_data(solver,output_space)
  normalise!(input,stats)
  normalise!(output,stats)
  return (input=merge_data(solver,input),output=output)
end

function fetch_data_from_stats(
  solver::NeuralSolver,
  input_space::InputSpace,
  stats::Normalisation
  )

  input = fetch_data(solver,input_space)
  normalise!(input,stats)
  return merge_data(solver,input)
end