function Algebra.solve(
  solver::NeuralSolver{A,<:DeepONetReduction},
  op::NeuralOperator,
  r::TransientRealisation,
  args...
  ) where A

  # Prepare input
  red = get_state_reduction(solver)
  strategy = get_strategy(red)
  V = get_test(op.op)
  coords0 = get_free_dof_coordinates(V)
  r_sampled = sample(strategy,r)
  params,coords = get_formatted_data(Float32,r_sampled,coords0)
  normalise!((params,coords),op.metadata)

  t = @timed begin
    pred_cpu = op.model((params,coords),op.metadata)
  end

  x̂ = _to_snapshots(pred_cpu,r)
  stats = CostTracker(t,nruns=num_params(r),name="DeepONet Transient Inference")

  return x̂,stats
end

function Algebra.solve(
  solver::NeuralSolver{A,<:NOMADReduction},
  op::NeuralOperator,
  r::TransientRealisation,
  args...
  ) where A

  # Prepare input
  red = get_state_reduction(solver)
  strategy = get_strategy(red)
  V = get_test(op.op)
  coords0 = get_free_dof_coordinates(V)
  r_sampled = sample(strategy,r)
  params,coords = get_formatted_data(Float32,r_sampled,coords0)
  pin,xin = _flatten(params,coords)
  normalise!((pin,xin),op.metadata)

  t = @timed begin
    pred_cpu = op.model((pin,xin),op.metadata)
  end

  x̂ = _to_snapshots(pred_cpu,r)
  stats = CostTracker(t,nruns= num_params(r),name="NOMAD Transient Inference")

  return x̂,stats
end

# utils 

function _to_snapshots(x,r)
  np = num_params(r)
  nt = num_times(r)
  d = reshape(permutedims(reshape(x,:,nt,np),(1,3,2)),:,nt*np)
  Snapshots(ConsecutiveParamArray(d),r)
end
