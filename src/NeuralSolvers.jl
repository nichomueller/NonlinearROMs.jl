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

RBSteady.get_state_reduction(solver::NeuralSolver) = solver.reduction
RBSteady.get_reduction(solver::NeuralSolver) = solver.reduction
get_sampler(solver::NeuralSolver) = get_sampler(get_reduction(solver))

"""
    struct NeuralOperator{O,T,A<:TrainedNeuralModel} <: ROMOperator{O,T}
      op::ParamOperator{O,T}
      model::A
    end

The evaluated Reduced Basis Operator for Neural Operators.
This struct is the direct output of the offline training phase and is passed to the `solve` function during the online phase.

It stores the high-fidelity operator, and the trained model (weights/states bundled inside it).

# Fields
- `op`: The original high-fidelity parametric operator.
- `model`: The trained [`TrainedNeuralModel`](@ref) (Lux chain + optimised parameters/states bundled together).
"""
struct NeuralOperator{O,T,A<:TrainedNeuralModel} <: ROMOperator{O,T}
  op::ParamOperator{O,T}
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
  feop::ParamOperator,
  s::AbstractSnapshots
  )

  model = train(solver,feop,s)
  NeuralOperator(feop,model)
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
base_strategy = NeuralReduction(model_arch,epochs=5000)
solver_base = NeuralSolver(LUSolver(),DeepONetReduction(base_strategy))
pretrained_op = reduced_operator(solver_base,feop,snapshots_base)

# Fine-Tuning with a smaller learning rate on a refined dataset
ft_strategy = NeuralReduction(
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
  model = train(reduction,feop,s,pretrained_op;update_stats=update_stats)
  NeuralOperator(feop,model)
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

function Algebra.solve(solver::NeuralSolver,op::NeuralOperator,r::AbstractRealisation)
  # Prepare input
  inputs = get_inputs(solver,op.op,r)
  normalise!(inputs,op.model.stats)
  formatted_inputs = prepare_inference_input(get_reduction(solver),inputs)
  
  # Inference
  t = @timed begin
    pred = first(op.model.chain(formatted_inputs,op.model.parameters,op.model.states))
    rescale!(pred,op.model.stats)
  end

  # Prepare output
  output = to_snapshots(pred,r)
  stats = CostTracker(t,nruns=num_params(r),name="Neural Operator Inference")

  return output,stats
end

# utils

prepare_inference_input(::NeuralReduction,inputs) = inputs

prepare_inference_input(::DeepONetReduction,(x,p)) = (p,x)
prepare_inference_input(::NOMADReduction,(x,p)) = (p,x)

prepare_inference_input(::AbstractKernelReduction,(x,p)) = tensor_of_coords(x,p)
function prepare_inference_input(red::GNOReduction,(x,p))
    # Get unified tensor
    tensor = tensor_of_coords(x,p)
    
    # Topology reconstruction
    graph = build_graph(DistanceGraph(red.model.radius),x)
    edge_index,edge_weights = get_edge_tensors(graph)
    
    return GraphData(tensor,edge_index,edge_weights)
end

function get_inputs_and_stats(args...;normalise=true)
  inputs = get_inputs(args...)
  stats = Normalisation(inputs...;normalise)
  return inputs,stats
end

function get_inputs(solver::NeuralSolver,op::ParamOperator,s::AbstractSnapshots)
  sampler = get_sampler(solver)
  sp = param_sample(sampler,s)
  data = get_param_data(sp)
  r = get_realisation(sp)
  trial = get_trial(op)
  coords = get_free_dof_coordinates(trial)
  d_mat,x_mat,p_mat = get_formatted_inputs(data,coords,r)
  sd_mat = sample(sampler.space_sampler,d_mat,1)
  sx_mat = sample(sampler.space_sampler,x_mat,1)
  return (sd_mat,sx_mat,p_mat)
end

function get_inputs(solver::NeuralSolver,op::ParamOperator,r::AbstractRealisation)
  sampler = get_sampler(solver)
  sr = sample(sampler,r)
  trial = get_trial(op)
  coords = get_free_dof_coordinates(trial)
  x_mat,p_mat = get_formatted_inputs(coords,sr)
  sx_mat = sample(sampler.space_sampler,x_mat,1)
  return (sx_mat,p_mat)
end

function get_formatted_inputs(args...)
  get_formatted_inputs(Float32,args...)
end

function get_formatted_inputs(::Type{T},values,coords::AbstractVector{<:Point},r::Realisation) where T
  d = T.(matrix_of_values(values))
  x = T.(matrix_of_coords(coords))
  p = T.(matrix_of_params(r))
  return (d,x,p)
end

function get_formatted_inputs(::Type{T},values,r::Realisation) where T
  d = T.(matrix_of_values(values))
  p = T.(matrix_of_params(r))
  return (d,p)
end

function get_formatted_inputs(::Type{T},values,coords::AbstractVector{<:Point},r::TransientRealisation) where T
  times = get_times(r)
  d = get_formatted_data(T,values,times)
  x = T.(matrix_of_coords(coords,times))
  p = T.(matrix_of_params(r))
  return (d,x,p)
end

function get_formatted_inputs(::Type{T},values,r::TransientRealisation) where T
  times = get_times(r)
  d = get_formatted_data(T,values,times)
  p = T.(matrix_of_params(r))
  return (d,p)
end

function get_formatted_inputs(::Type{T},coords::AbstractVector{<:Point},r::AbstractRealisation) where T
  x = T.(matrix_of_coords(coords))
  p = T.(matrix_of_params(r))
  return (x,p)
end

function get_formatted_inputs(::Type{T},coords::AbstractVector{<:Point},r::TransientRealisation) where T
  x = T.(matrix_of_coords(coords,get_times(r)))
  p = T.(matrix_of_params(r))
  return (x,p)
end

function get_formatted_inputs(::Type{T},coords::AbstractVector{<:Point},r::Realisation) where T
  x = T.(matrix_of_coords(coords))
  p = T.(matrix_of_params(r))
  return (x,p)
end

function to_snapshots(x,r::Realisation)
  np = num_params(r)
  d = reshape(x,:,np)
  Snapshots(ConsecutiveParamArray(d),r)
end

function to_snapshots(x,r::TransientRealisation)
  np = num_params(r)
  nt = num_times(r)
  d = reshape(permutedims(reshape(x,:,nt,np),(1,3,2)),:,nt*np)
  Snapshots(ConsecutiveParamArray(d),r)
end
