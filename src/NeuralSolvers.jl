"""
    const NeuralSolver{A,B<:NeuralReduction} = GlobalRBSolver{A,B,Nothing,Nothing}

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

strategy = NeuralStrategy(
  DeepONet(2,2;width=128,depth=4,activation=Lux.gelu),
  epochs = 1000
  )
reduction = DeepONetReduction(strategy)
solver = NeuralSolver(ThetaMethod(LUSolver(),dt,θ),reduction)
```
"""

struct NeuralSolver{A,B<:NeuralReduction} <: ROMSolver
  fesolver::A
  reduction::B
end

"""
    struct NeuralOperator{O,T,A<:TrainedNeuralModel,B} <: ROMOperator{O,T}
      op::ParamOperator{O,T}
      model::A
      metadata::B
    end

The evaluated Reduced Basis Operator for Neural Operators.
This struct is the direct output of the offline training phase and is passed to the `solve` function during the online phase.

It stores the high-fidelity operator, the trained model (weights/states bundled inside it),
and any normalization metadata needed to scale the data.

# Fields
- `op`: The original high-fidelity parametric operator.
- `model`: The trained [`TrainedNeuralModel`](@ref) (Lux chain + optimised parameters/states bundled together).
- `metadata`: Either `identity` (no normalisation) or a [`NormStats`](@ref) bundling the
  z-score statistics used to normalize the inputs and the absolute maximum scalar value of
  the snapshot target data (`metadata.dmax`), used for the final denormalization of the
  network predictions.
"""
struct NeuralOperator{O,T,A<:TrainedNeuralModel,B} <: ROMOperator{O,T}
  op::ParamOperator{O,T}
  model::A
  metadata::B
end

function NeuralOperator(op,model)
  NeuralOperator(op,model,identity)
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
  feop::ParamOperator,
  s::AbstractSnapshots
  )

  reduction = get_state_reduction(solver)
  model,metadata... = train(reduction,feop,s)
  NeuralOperator(feop,model,metadata...)
end

"""
    reduced_operator(
      solver::NeuralSolver,
      feop::ParamOperator,
      s::AbstractSnapshots,
      pretrained_op::NeuralOperator;
      update_stats::Bool=false
    )

Performs **Fine-Tuning (Continual or Transfer Learning)** on a previously trained Neural Operator.
It initializes the neural network with the weights and states of the `pretrained_op`, continuing the training using the newly provided snapshots `s` and the configuration defined in `solver`.

# Arguments
- `solver`: The `NeuralSolver` containing the updated training configuration (e.g., lower learning rate, new epochs).
- `feop`: The high-fidelity parametric operator.
- `s`: The new `Snapshots` dataset for fine-tuning.
- `pretrained_op`: The previously trained `NeuralOperator`.

# Keyword Arguments
- `update_stats::Bool`: Dictates how data normalization is handled.
  - If `false` (default): The model inherits the original normalization statistics (\$\\mu\$, \$\\sigma\$, and `max_u`) from the `pretrained_op`. Best for **Continual Learning** where the new data is drawn from the same underlying distribution.
  - If `true`: The model recomputes entirely new normalization statistics based solely on the new snapshots `s`. Best for **Transfer Learning** when shifting to a drastically different parameter space or domain scale.

# Examples
```julia
# Define the shared architecture (2 params -> Branch; 2D coords -> Trunk)
model_arch = DeepONet(2,2)

# Base Training
base_strategy = NeuralStrategy(model_arch,epochs=5000)
solver_base = NeuralSolver(LUSolver(),DeepONetReduction(base_strategy))
pretrained_op = reduced_operator(solver_base,feop,snapshots_base)

# Fine-Tuning with a smaller learning rate on a refined dataset
ft_strategy = NeuralStrategy(
  model_arch, # match the pretrained one
  epochs = 1000,
  lr_scheduler = CosineAnnealing(1000,lr_max=1e-5) # Smaller LR
  )
solver_ft = NeuralSolver(LUSolver(),DeepONetReduction(ft_strategy))

# Continual learning (inherits original stats)
new_op = reduced_operator(solver_ft,feop,snapshots_new,pretrained_op;update_stats=false)
```
"""
function RBSteady.reduced_operator(
  solver::NeuralSolver,
  feop::ParamOperator,
  s::AbstractSnapshots,
  pretrained_op::NeuralOperator;
  update_stats::Bool=false
  )

  reduction = get_state_reduction(solver)
  model,metadata... = train(reduction,feop,s,pretrained_op;update_stats=update_stats)
  NeuralOperator(feop,model,metadata...)
end

"""
    reduced_operator(
      solver::NeuralSolver,
      s::AbstractSnapshots,
      pretrained_op::NeuralOperator;
      update_stats::Bool=false
    )

Automatically extracts the high-fidelity operator (`feop`) from `pretrained_op.op` and invokes the main fine-tuning routine.
"""
function RBSteady.reduced_operator(
  solver::NeuralSolver,
  s::AbstractSnapshots,
  pretrained_op::NeuralOperator;
  update_stats::Bool=false
  )

  feop = pretrained_op.op
  reduced_operator(solver,feop,s,pretrained_op;update_stats=update_stats)
end

function Algebra.solve(
  solver::NeuralSolver{A,<:KernelOperatorReduction},
  op::NeuralOperator,
  r::Realisation
  ) where A

  # Prepare input
  red = get_state_reduction(solver)
  strategy = get_strategy(red)
  coords = get_free_dof_coordinates(get_test(op.op))
  
  r_sampled = sample(strategy,r)
  params,coords = get_formatted_data(Float32,r_sampled,coords)
  
  # Normalize inputs using training metadata prior to concatenation
  normalise!((params,coords),op.metadata)
  
  # Build the 3D tensor expected by the Lifting Layer
  input_tensor = tensor_of_coords(coords,params)

  # Inference (denormalizes the output internally via the metadata fallback)
  t = @timed begin
    pred_cpu = op.model(input_tensor,op.metadata)
  end

  # Reshaping the [out_channels, N_nodes, Batch] output back to Snapshots format (N_dofs, n_samples)
  out_channels = size(pred_cpu,1)
  n_nodes = size(coords,2)
  n_samples = size(params,2)
  pred_2d = reshape(pred_cpu,out_channels * n_nodes,n_samples)

  x̂ = Snapshots(ConsecutiveParamArray(pred_2d),r)
  stats = CostTracker(t,nruns=num_params(r),name="Kernel Operator Inference")

  return x̂,stats
end

function Algebra.solve(
  solver::NeuralSolver{A,<:DeepONetReduction},
  op::NeuralOperator,
  r::Realisation
  ) where A

  # Prepare input
  red = get_state_reduction(solver)
  strategy = get_strategy(red)
  coords = get_free_dof_coordinates(get_test(op.op))
  r_sampled = sample(strategy,r)
  params,coords = get_formatted_data(Float32,r_sampled,coords)
  normalise!((params,coords),op.metadata)

  # Inference Execution (denormalizes the output internally, using op.metadata.dmax)
  t = @timed begin
    pred_cpu = op.model((params,coords),op.metadata)
  end

  x̂ = Snapshots(ConsecutiveParamArray(pred_cpu),r)
  stats = CostTracker(t,nruns=num_params(r),name="DeepONet Inference")

  return x̂,stats
end

function Algebra.solve(
  solver::NeuralSolver{A,<:NOMADReduction},
  op::NeuralOperator,
  r::Realisation
  ) where A

  # Prepare input
  red = get_state_reduction(solver)
  strategy = get_strategy(red)
  coords = get_free_dof_coordinates(get_test(op.op))
  r_sampled = sample(strategy,r)
  params,coords = get_formatted_data(Float32,r_sampled,coords)
  pin,xin = _flatten(params,coords)
  normalise!((pin,xin),op.metadata)

  # Inference (denormalizes the output internally, using op.metadata.dmax)
  t = @timed begin
    pred_cpu = op.model((pin,xin),op.metadata)
  end

  # Reshaping of the output for GridapROMs (N_dofs,n_samples)
  pred_2d = reshape(pred_cpu,size(coords,2),size(params,2))

  x̂ = Snapshots(ConsecutiveParamArray(pred_2d),r)
  stats = CostTracker(t,nruns=num_params(r),name="NOMAD Inference")

  return x̂,stats
end
