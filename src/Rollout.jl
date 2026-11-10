function Base.collect(sol::TransientParamSolution)
  r = sol.odeosol.r
  vals = collect(sol.odesol)
  trial = get_trial(sol.odesol.odeop)
  FEFunction(trial(r),vals)
end

"""
    rollout_data(fspace,features_fn::Function,times,states;noise_std=TData(0.01),rng=Random.default_rng())

Assembles one-step autoregressive training data `(x,features(t),u(t)) -> u(t+dt)-u(t)`
from a collected transient trajectory (`times[i]`/`states[i]` is the state at time
`times[i]`, `states[1]` being the given initial condition -- see `collect_trajectory`,
which only collects the *solved* steps, not `t0` itself). Every consecutive pair
`(t_i,t_{i+1})`, for every parameter already batched into `states`, becomes one
training sample, stacked along the batch axis: an `Nt`-step, `K`-parameter trajectory
yields `(Nt-1)*K` samples, not just `K` -- the main reason autoregressive training is
so much more data-efficient than fitting a single space-time map.

`features_fn(t) -> FeatureBundle` supplies whatever time-dependent data (besides the
running state `u(t)`) the kernel should see, e.g.
`t -> FeatureBundle((aμt(μ,t),fμt(μ,t),gμt(μ,t),hμt(μ,t)))` for `test/heat_equation.jl`;
it's evaluated once per time step, not per parameter (parameters are already batched
into `times`/`states`/`FeatureBundle` itself).

Predicting the *change* `u(t+dt)-u(t)` rather than the absolute next state is the
standard choice for autoregressive PDE surrogates: it's a smaller, more stationary
target (the "do nothing" baseline is already close to right), which eases
optimisation and keeps `MScore`'s single global `max` from being dominated by
whichever field has the largest absolute magnitude rather than the largest *change*.
`rollout` undoes this (`u(t+dt) = u(t) + predicted_change`) automatically.

`noise_std`: relative Gaussian noise added to the state channel at training time
(`u(t_i) + noise_std * maximum(abs,u(t_i)) * randn()`, independently per sample),
*not* to the other time-dependent features -- a simplified stand-in for Brandstetter
et al.'s "pushforward trick" (training on the model's own rolled-out predictions
rather than only ground truth). Since the model only ever sees exact states during
training but its own imperfect predictions during rollout, training it to also
correct a perturbed input makes it more robust to that distribution shift and
reduces error accumulation over long rollouts. Set to `0` to disable.
"""
function rollout_data(
  fspace,
  features_fn::Function,
  times::AbstractVector,
  states::AbstractVector;
  noise_std::Real=TData(0.01),
  rng::AbstractRNG=Random.default_rng()
  )

  Nt = length(times)
  @check Nt == length(states) "Need exactly one state per time"
  @check Nt > 1 "Need at least 2 time steps to form a consecutive pair"

  x = get_coords(fspace)
  rid1 = _get_values(states[1],x)
  ranges = rid1.ranges
  point_ranges = rid1.point_ranges

  ins = Vector{AbstractInputData}(undef,Nt-1)
  outs = Vector{AbstractInputData}(undef,Nt-1)
  for i in 1:Nt-1
    feat_i = _get_features(features_fn(times[i]),x)
    u_i = _get_values(states[i],x).data
    u_ip1 = _get_values(states[i+1],x).data
    if !iszero(noise_std)
      u_i = u_i .+ TData(noise_std).*maximum(abs,u_i).*TData.(randn(rng,size(u_i)))
    end
    ins[i] = vcat(feat_i,u_i)
    outs[i] = u_ip1 .- u_i
  end

  input = cat(ins...;dims=3)
  output = cat(outs...;dims=3)
  coords = _get_coords(x,size(input,3))

  input_space = InputSpace(coords,input)
  output_space = OutputSpace(get_fe_space(states[1]),InputData(output,ranges,point_ranges))
  return input_space,output_space
end

"""
    rollout(solver,op,fspace,features_fn::Function,times,u0) -> Vector{<:AbstractInputData}

Autoregressively applies the trained one-step operator `op` (predicting the
*change* `u(t+dt)-u(t)`, see `rollout_data`) starting from `u0` -- either the
discretised initial state directly (same point/channel layout as `rollout_data`'s
state channel) or a `ParamFEFunction` (e.g. `states[1]` from `collect_trajectory`,
or the initial condition itself) -- producing the full predicted trajectory
`[u0,u(t_2),...,u(t_end)]`; `times[1]`'s state is `u0` itself, so no solve happens
for it.

Each step only ever sees its OWN previous prediction, never the true trajectory, so
whatever drift the model accumulates over the horizon shows up here, not just in
next-step validation error -- this is the function to use for assessing
longer-horizon rollout accuracy, not just one-step accuracy.
"""
function rollout(
  solver::NeuralSolver,
  op::NeuralOperator,
  fspace,
  features_fn::Function,
  times::AbstractVector,
  u0::AbstractInputData
  )

  x = get_coords(fspace)
  K = size(u0,3)
  states = Vector{AbstractInputData}(undef,length(times))
  states[1] = u0
  u = u0
  for i in 1:length(times)-1
    feat_i = _get_features(features_fn(times[i]),x)
    @check size(feat_i,3) == K "features_fn(t) must be evaluated at the same K parameters as u0"
    input_i = InputSpace(_get_coords(x,K),vcat(feat_i,u))
    du = solve(solver,op,input_i)
    u = u .+ du
    states[i+1] = u
  end
  return states
end

function rollout(solver::NeuralSolver,op::NeuralOperator,fspace,features_fn::Function,times,u0::ParamFEFunction)
  x = get_coords(fspace)
  u0_vals = _get_values(u0,x).data
  rollout(solver,op,fspace,features_fn,times,u0_vals)
end
