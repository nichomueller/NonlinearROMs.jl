"""
    module Rollout (included into NonlinearROMs)

Autoregressive one-step training/inference for transient problems: instead of
learning a single map `parameters -> u(T)` (or a full space-time operator), we learn
one *time-stepping* operator `(x,a(x,t),...,u(t)) -> u(t+dt)-u(t)` and apply it
repeatedly. This is the natural choice when the underlying PDE's physics is
time-invariant (the same map governs every step): every consecutive pair in every
trajectory becomes one training sample, so a handful of trajectories already gives
plenty of data, and the architecture/training machinery is exactly the existing
steady-state one (`KernelNeuralModel`, `NeuralSolver`, `neural_operator`,
`Algebra.solve`) -- the only difference is what `InputSpace`/`OutputSpace` hold.

The known failure mode of this approach is rollout drift: the model only ever sees
*exact* states as input during training, but at inference it's fed its own
(slightly imperfect) predictions, so small per-step errors can compound over a long
rollout. `rollout_data`'s `noise_std` (a simplified stand-in for Brandstetter et
al.'s "pushforward trick") perturbs the training-time state input to make the model
more robust to that distribution shift; `rollout` is also the function to use to
actually measure how much drift remains over a given horizon, since next-step
validation error alone won't show it.

Usage (see `test/heat_equation.jl`):
```julia
times = collect(tdomain)
states = [uh0μ(μ);collect_trajectory(solve(fesolver,feop,μt,uh0μ))]
features_fn(t) = FeatureBundle((aμt(μ,t),fμt(μ,t),gμt(μ,t),hμt(μ,t)))
input_space,output_space = rollout_data(fspace,features_fn,times,states)

model = KernelNeuralModel(gspace;hidden_layers=(16,16),activation=tanh)
solver = NeuralSolver(model;epochs=200,print_every=50)
op = neural_operator(solver,input_space,output_space)

predicted_states = rollout(solver,op,fspace,features_fn,times,states[1])
```
"""

"""
    collect_trajectory(sol) -> states::Vector

Eagerly collects a transient Gridap/GridapROMs solve's iterator (as returned by
`solve(fesolver,feop,μt,uh0μ)` for a `TransientLinearParamFEOperator`) into memory.
Autoregressive training needs random access to every consecutive pair across the
whole trajectory, not just a single forward pass over it, and the iterator is
typically not cheaply re-iterable (each step depends on solving from the previous
one).

Yields one state per *solved* step -- i.e. `t0` (the given initial condition) is
excluded, since the iterator never solves for it; prepend it (and its own known
time) yourself before calling `rollout_data`/`rollout`, e.g.
`times = collect(tdomain); states = [uh0μ(μ); collect_trajectory(uhμ)]`
(see `test/heat_equation.jl`).
"""
function collect_trajectory(sol)
  states = []
  for uh in sol
    push!(states,uh)
  end
  identity.(states)
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
