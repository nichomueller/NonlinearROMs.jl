"""
    struct NeuralSolver{A<:NeuralModel,B} <: ROMSolver
      fesolver::B
      reduction::NeuralReduction{A}
    end

    NeuralSolver(fesolver::GridapType,reduction::NeuralReduction)

Initializes the Reduced Basis Solver for Neural Operators.

# Arguments
- `fesolver`: The high-fidelity standard Gridap solver (e.g., `LUSolver()`). In the context of Reduced Order Models, the neural operator acts as a surrogate for this specific full-order solver. This reference defines the underlying high-fidelity model being approximated.
- `reduction::NeuralReduction`: The configured neural reduction strategy (e.g., `DeepONetReduction` or `NOMADReduction`).

# Examples

**Minimal Default Initialization:**
```julia
# Default hyperparameters (20000 epochs, full-batch, etc.), 2 params -> Branch, 2D coords -> Trunk
solver = NeuralSolver(LUSolver(),DeepONetReduction(model=DeepONet(2,2)))
```
**Custom Initialization:**
```julia
using Lux

reduction = NeuralReduction(
  DeepONet(2,2;width=128,depth=4,activation=Lux.gelu),
  epochs = 1000
)
solver = NeuralSolver(ThetaMethod(LUSolver(),dt,θ),reduction)
```
"""
struct NeuralSolver{A<:NeuralModel,B} <: ROMSolver
  fesolver::B
  reduction::NeuralReduction{A}
end

function NeuralSolver(fesolver::GridapType,model::NeuralModel;kwargs...)
  reduction = NeuralReduction(model;kwargs...)
  NeuralSolver(fesolver,reduction)
end

get_reduction(s::NeuralSolver) = s.reduction

get_formatter(s::NeuralSolver) = get_formatter(s.reduction)
get_optimiser(s::NeuralSolver) = get_optimiser(s.reduction)
get_scheduler(s::NeuralSolver) = get_scheduler(s.reduction)
get_logger(s::NeuralSolver) = get_logger(s.reduction)
build_model(s::NeuralSolver,args...;kwargs...) = build_model(s.reduction,args...;kwargs...)
resolve_batch_size(s::NeuralSolver,inputs) = resolve_batch_size(s.reduction,inputs)

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

# utils

fetch_data(solver::NeuralSolver,args...) = fetch_data(get_formatter(solver),args...)
fetch_stats(solver::NeuralSolver,args...) = fetch_stats(get_formatter(solver),args...)
merge_data(solver::NeuralSolver,args...) = merge_data(get_formatter(solver),args...)

function fetch_data_and_stats(
  solver::NeuralSolver,
  input_space::InputSpace,
  output_space::OutputSpace
  )

  data = (
    input = fetch_data(solver,input_space),
    output = fetch_data(solver,output_space)
  )
  stats = fetch_stats(solver,data)
  mdata = merge_data(solver,data)
  return mdata,stats
end

function fetch_data_and_stats(
  solver::NeuralSolver,
  input_space::InputSpace
  )

  data = (
    input = fetch_data(solver,input_space),
  )
  stats = fetch_stats(solver,data)
  mdata = merge_data(solver,data)
  return mdata,stats
end

function fetch_data_from_stats(
  solver::NeuralSolver,
  input_space::InputSpace,
  output_space::OutputSpace,
  stats::Normalisation
  )

  data = (
    input = fetch_data(solver,input_space),
    output = fetch_data(solver,output_space)
  )
  normalise!(data,stats)
  mdata = merge_data(solver,data)
  return mdata
end

function fetch_data_from_stats(
  solver::NeuralSolver,
  input_space::InputSpace,
  stats::Normalisation
  )

  data = (
    input = fetch_data(solver,input_space),
  )
  normalise!(data,stats)
  mdata = merge_data(solver,data)
  return mdata
end