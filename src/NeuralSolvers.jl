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
    struct NeuralOperator{O,A<:TrainedNeuralModel} <: ROMOperator{O}
      op::ParamOperator{O}
      model::A
    end

The evaluated Reduced Basis Operator for Neural Operators.
This struct is the direct output of the offline training phase and is passed to the `solve` function during the online phase.

It stores the high-fidelity operator, and the trained model (weights/states bundled inside it).

# Fields
- `op`: The original high-fidelity parametric operator.
- `model`: The trained [`TrainedNeuralModel`](@ref) (Lux chain + optimised parameters/states bundled together).
"""
struct NeuralOperator{O,A<:TrainedNeuralModel} <: ROMOperator{O}
  op::ParamOperator{O}
  model::A
end

ParamSteady.get_fe_operator(op::NeuralOperator) = op.op

"""
    reduced_operator(
      solver::NeuralSolver,
      feop::ParamOperator,
      s::AbstractSnapshots
    )

Executes the **Offline Phase** for Neural Operators on steady-state problems.
This method triggers the training loop of the neural network specified in the `solver`.

It automatically extracts the training dataset (parameters/sensors and spatial coordinates) from the snapshots `s` and the FE operator `feop`, normalizes the data, and performs the optimization.

Returns a `NeuralOperator` containing the trained network, its optimized weights, and the normalization statistics required for the online phase.
"""
function RBSteady.reduced_operator(
  solver::NeuralSolver,
  op::ParamOperator,
  s::AbstractSnapshots;
  kwargs...
  )

  model = train(solver,op,s;kwargs...)
  NeuralOperator(op,model)
end

function Algebra.solve(solver::NeuralSolver,op::NeuralOperator,r::AbstractRealisation)
  # Prepare input
  data = fetch_data_from_stats(solver,op,r)
  # Inference
  output = op.model(data)
  return output
end

# utils

function fetch_data_and_stats(
  solver::NeuralSolver,
  r::AbstractRealisation,
  args...
  )

  formatter = get_formatter(solver)
  space = get_discrete_space(solver)
  feats = get_features(solver,r)
  fetch_data_and_stats(formatter,space,feats,args...)
end

function fetch_data_from_stats(
  solver::NeuralSolver,
  op::NeuralOperator,
  r::AbstractRealisation,
  args...
  )

  formatter = get_formatter(solver)
  space = get_discrete_space(solver)
  feats = get_features(solver,r)
  stats = get_stats(op)
  fetch_data_from_stats(stats,formatter,space,feats,args...)
end