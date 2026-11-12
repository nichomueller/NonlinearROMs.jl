# function Base.collect(sol::TransientParamFESolution)
#   r = sol.odesol.r
#   vals = collect(sol.odesol)
#   trial = sol.trial(r)
#   FEFunction(trial,vals)
# end

# Gridap.FESpaces.get_fe_space(sol::TransientParamFESolution) = sol.trial(sol.odesol.r)

"""
    struct RolloutData{A,B<:AbstractVector,C<:Union{Tuple,AbstractVector}}
      data::A
      times::B
      steps::C
    end

Temporal-bundling layout (Brandstetter et al.) for an autoregressive neural
operator `N(u^i) -> (u^{i+1},...,u^{i+K})`, `K` the cluster size. `steps` cuts
`times` (indices into it, not time values) into clusters of `K+1` consecutive
time indices each (`steps[k]:steps[k+1]`): index 1 is the input time `i`,
indices `2:K+1` are the `K` target times `i+1,...,i+K`.
"""
struct RolloutData{A,B<:AbstractVector,C<:Union{Tuple,AbstractVector}}
  data::A
  times::B
  steps::C
end

function RolloutData(data,times::AbstractVector,step::Int=2)
  @check step % 2 == 0 "Step must be even"
  steps = 1:step:length(times)
  last(steps) != length(times) && @warn "Removing $((times[last(steps)+1],times[end])) from rollout: incomplete cluster"
  RolloutData(data,times,steps)
end

# Gridap.FESpaces.get_fe_space(a::RolloutData) = get_fe_space(a.data)

function Base.collect(a::RolloutData)
  data = collect(a.data)
  T = typeof(data)
  rollout = Vector{T}(undef,length(a.steps)-1)
  for k in eachindex(rollout)
    rollout[k] = data[a.steps[k]:a.steps[k+1]]
  end
  return rollout
end

function Base.collect(a::RolloutData{<:TransientParamFESolution})
  _collect_trajectory(a.data,a.steps)
end

# function Base.collect(a::RolloutData{<:TransientParamFESolution})
#   states = [xh for (_,xh) in a.data]
#   T = eltype(states)
#   rollout = Vector{Vector{T}}(undef,length(a.steps)-1)
#   for k in eachindex(rollout)
#     rollout[k] = states[a.steps[k]:a.steps[k+1]]
#   end
#   rollout
# end

function Base.collect(a::RolloutData{<:Function})
  map(1:length(a.steps)-1) do k
    ts = a.times[a.steps[k]:a.steps[k+1]]
    [a.data(t) for t in ts]
  end
end

# the regular (non-ragged) cluster size K, and the clusters that actually have it
function _full_clusters(clusters::AbstractVector)
  K = maximum(c -> length(c)-1,clusters)
  filter(c -> length(c)-1 == K,clusters)
end

"""
    InputSpace(x::AbstractVector{<:Point},a::RolloutData) -> InputSpace

Builds the fixed (non-autoregressive) conditioning data for one cluster/sample:
the features at every one of its `K+1` times (input time included), channel-
concatenated, so the kernel can see the forcing/BC values at every time it
must predict. The autoregressive state `u^i` itself is *not* part of this --
it is threaded separately through the model as its running hidden state (`v`,
Kovachki et al.'s `pass_new_data`), exactly as in the steady case.
"""
function InputSpace(x::AbstractVector{<:Point},a::RolloutData)
  clusters = _full_clusters(collect(a))
  per_cluster = map(c -> _get_features(c,x),clusters)
  features = cat(per_cluster...;dims=3)
  coords = points_to_coords(x,size(features,3))
  InputSpace(coords,features)
end

function InputSpace(space::MultiFieldFESpace,a::RolloutData)
  x = get_coords(space)
  fields = Tuple(space.spaces)
  gspace = InputSpace(x,a)
  lspaces = map(s -> InputSpace(s,a),fields)
  l_to_lcoord_to_coords = _lcoord_to_coords_map(map(get_coords,fields),x)
  MultipleSpaces(gspace,lspaces,l_to_lcoord_to_coords,nothing)
end

# per cluster: drop the input state, stack the K target states channel-wise;
# across clusters: stack along the sample axis. `getstate` extracts the field
# of interest out of a (possibly multi-field) per-time-step state
function _rollout_values(a::RolloutData,x::AbstractVector{<:Point},getstate=identity)
  clusters = _full_clusters(collect(a))
  per_cluster = map(clusters) do states
    targets = @views states[2:end]
    cat(map(s -> _get_values(getstate(s),x),targets)...;dims=1)
  end
  cat(per_cluster...;dims=3)
end

"""
    OutputSpace(uh::RolloutData{<:TransientParamFESolution}) -> OutputSpace

Builds the autoregressive target data for one cluster/sample: the `K` *future*
states `u^{i+1},...,u^{i+K}` (the input state `u^i` is excluded -- it belongs to
`InputSpace`'s pass-through state, not to the target), channel-concatenated
into a `(D*K,N,P)` array, one sample per (original parameter, cluster) pair.
"""
function OutputSpace(uh::RolloutData{<:TransientParamFESolution})
  space = get_fe_space(uh)
  _rollout_output_space(space,uh)
end

function _rollout_output_space(space::SingleFieldFESpace,uh::RolloutData)
  x = get_coords(space)
  values = _rollout_values(uh,x)
  OutputSpace(space,values)
end

function _rollout_output_space(space::MultiFieldFESpace,uh::RolloutData)
  x = get_coords(space)
  lspaces = ntuple(length(space.spaces)) do i
    xi = get_coords(space.spaces[i])
    values_i = _rollout_values(uh,xi,s -> s[i])
    OutputSpace(space.spaces[i],values_i)
  end
  l_to_lcoord_to_coords = _lcoord_to_coords_map(map(get_coords,lspaces),x)
  values = cat_channels(map(get_values,lspaces),l_to_lcoord_to_coords,length(x))
  MultipleSpaces(OutputSpace(space,values),lspaces,l_to_lcoord_to_coords,nothing)
end

function OutputSpace(sol::TransientParamFESolution,args...)
  times = get_times(sol.odesol.r)
  data = RolloutData(sol,times,args...)
  OutputSpace(data)
end

# geometry (the integration measure/graph) doesn't depend on the time-clustering
FunctionSpace(a::RolloutData) = FunctionSpace(get_fe_space(a))
GraphInducedSpace(a::RolloutData;kwargs...) = GraphInducedSpace(get_fe_space(a);kwargs...)

# function rollout_data(
#   fspace,
#   features_fn::Function,
#   times::AbstractVector,
#   states::AbstractVector;
#   noise_std::Real=TData(0.01),
#   rng::AbstractRNG=Random.default_rng()
#   )

#   Nt = length(times)
#   @check Nt == length(states) "Need exactly one state per time"
#   @check Nt > 1 "Need at least 2 time steps to form a consecutive pair"

#   x = get_coords(fspace)
#   rid1 = _get_values(states[1],x)
#   ranges = rid1.ranges
#   point_ranges = rid1.point_ranges

#   ins = Vector{AbstractInputData}(undef,Nt-1)
#   outs = Vector{AbstractInputData}(undef,Nt-1)
#   for i in 1:Nt-1
#     feat_i = _get_features(features_fn(times[i]),x)
#     u_i = _get_values(states[i],x).data
#     u_ip1 = _get_values(states[i+1],x).data
#     if !iszero(noise_std)
#       u_i = u_i .+ TData(noise_std).*maximum(abs,u_i).*TData.(randn(rng,size(u_i)))
#     end
#     ins[i] = vcat(feat_i,u_i)
#     outs[i] = u_ip1 .- u_i
#   end

#   input = cat(ins...;dims=3)
#   output = cat(outs...;dims=3)
#   coords = points_to_coords(x,size(input,3))

#   input_space = InputSpace(coords,input)
#   output_space = OutputSpace(get_fe_space(states[1]),InputData(output,ranges,point_ranges))
#   return input_space,output_space
# end

# function rollout(
#   solver::NeuralSolver,
#   op::NeuralOperator,
#   fspace,
#   features_fn::Function,
#   times::AbstractVector,
#   u0::AbstractInputData
#   )

#   x = get_coords(fspace)
#   K = size(u0,3)
#   states = Vector{AbstractInputData}(undef,length(times))
#   states[1] = u0
#   u = u0
#   for i in 1:length(times)-1
#     feat_i = _get_features(features_fn(times[i]),x)
#     @check size(feat_i,3) == K "features_fn(t) must be evaluated at the same K parameters as u0"
#     input_i = InputSpace(points_to_coords(x,K),vcat(feat_i,u))
#     du = solve(solver,op,input_i)
#     u = u .+ du
#     states[i+1] = u
#   end
#   return states
# end

# function rollout(solver::NeuralSolver,op::NeuralOperator,fspace,features_fn::Function,times,u0::ParamFEFunction)
#   x = get_coords(fspace)
#   u0_vals = _get_values(u0,x).data
#   rollout(solver,op,fspace,features_fn,times,u0_vals)
# end

# utils

for (F,T) in zip((:_get_features,:_get_values),(:FeatureBundle,:ParamFEFunction))
  @eval function $F(data::AbstractVector{<:$T},x::AbstractVector{<:Point})
    ax = map(data) do a
      $F(a,x)
    end
    cat(ax...;dims=1)
  end
end

function _collect_trajectory(fesol::TransientParamFESolution,steps)
  sol = fesol.odesol
  u0 = first(sol.us0)
  nsteps = length(steps)
  step = steps[2] - steps[1]
  ncols = num_params(sol.r)*step
  cache = Vector{typeof(u0)}(undef,nsteps)
  for (k,(rk,uk)) in enumerate(sol)
    k > last(steps) && break
    sols = ParamODEs._allocate_solutions(u0,ncols)
    ParamODEs._collect_solutions!(sols,uk,k)
    if (k+1) in steps
      cache[k] = sols
      sols = ParamODEs._allocate_solutions(u0,ncols)
    end
  end
  return cache
end

function _collect_trajectory(sol::ODEParamSolution,steps)
  msg = """ 
  The values should be consecutive for efficiency. Deactivate 
  BlockMultiFieldStyle() in the MultiFieldFESpace!
  """
  @notimplemented msg
end

function _collect_trajectory(sol::ODEParamSolution{<:ConsecutiveParamVector},steps)
  u0 = first(sol.us0)
  nsteps = length(steps)
  step = steps[2] - steps[1]
  ncols = num_params(sol.r)*step
  cache = Vector{typeof(u0)}(undef,nsteps)
  for (k,(rk,uk)) in enumerate(sol)
    k > last(steps) && break
    sols = ParamODEs._allocate_solutions(u0,ncols)
    ParamODEs._collect_solutions!(sols,uk,k)
    if (k+1) in steps
      cache[k] = sols
      sols = ParamODEs._allocate_solutions(u0,ncols)
    end
  end
  return cache
end