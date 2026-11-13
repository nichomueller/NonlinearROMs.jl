struct TwoStageBuckets{A<:AbstractInputData}
  first_stage::Vector{A}
  second_stage::Vector{A}
end

function TwoStageBuckets(data,steps::Union{Tuple,AbstractVector})
  r1 = stage_ranges(steps;stage=:first)
  r2 = stage_ranges(steps;stage=:second)
  s1,s2 = split_stages(data,r1,r2)
  TwoStageBuckets(s1,s2)
end

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

function TwoStageBuckets(a::RolloutData)
  TwoStageBuckets(a,a.steps)
end

function stage_ranges(a::RolloutData;kwargs...)
  stage_ranges(a.steps;kwargs...)
end

function Base.collect(a::RolloutData{<:TransientParamFESolution})
  sol = a.data
  μ = get_params(sol.odesol.r)
  vals = collect(sol.odesol)
  np = length(μ)
  map(1:length(a.steps)-1) do k
    ik = a.steps[k]:a.steps[k+1]
    ipk = (a.steps[k]-1)*np+1:a.steps[k+1]*np
    tk = a.times[ik]
    vk = @views vals[ipk] 
    trial = sol.trial(μ,tk)
    FEFunction(trial,vk)
  end
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

function stage_ranges(steps;stage=:first)
  t = ()
  for k in 1:length(steps)-1
    ak = steps[k]
    bk = steps[k+1]
    mk = div((ak - 1) + bk,2)
    if stage==:first
      t = (t...,ak:mk)
    elseif stage==:second
      t = (t...,mk+1:bk)
    else
      @notimplemented "stage must be :first or :second"
    end
  end
  return t
end

function split_stages(a,r1,r2)
  @check length(r1) == length(r2) "r1 and r2 must have the same length"
  _split_stages(a,r1,r2)
end

function _split_stages(a,r1,r2)
  @abstractmethod
end

function _split_stages(a::RolloutData,r1,r2)
  data = collect(a.data)
  T = typeof(data)
  stage1 = Vector{T}(undef,length(r1)-1)
  stage2 = Vector{T}(undef,length(r2)-1)
  for k in eachindex(r1)
    stage1[k] = data[r1[k]:r1[k+1]]
    stage2[k] = data[r2[k]:r2[k+1]]
  end
  return stage1,stage2
end

function _split_stages(a::RolloutData{<:FeatureBundle},r1,r2,x::AbstractVector{<:Point})
  @check length(r1) == length(r2) "r1 and r2 must have the same length"
  np = num_params(a.data)
  features,offsets = _init_data(a,x)

  T = typeof(features)
  stage1 = Vector{T}(undef,length(r1))
  stage2 = Vector{T}(undef,length(r2))
  for k in eachindex(r1)
    stage1[k] = similar(features)
    stage2[k] = similar(features)
  end

  Δ = length(first(r1))
  
  @inbounds for (j,aj) in enumerate(a.data.bundle)
    ax = lazy_map(aj,x)
    cache = array_cache(ax)
    for i in eachindex(ax)
      axi = getindex!(cache,ax,i)
      for k in eachindex(r1)
        o = offsets[j+(k-1)*Δ]
        _fill_stage!(stage1[k],axi,o,r1[k],i,np)
        _fill_stage!(stage2[k],axi,o,r2[k],i,np)
      end
    end
  end

  stage1,stage2
end

function _split_stages(a::RolloutData{<:TransientParamFESolution},r1,r2,x::AbstractVector{<:Point})
  @check length(r1) == length(r2) "r1 and r2 must have the same length"
  np = num_params(a.data)
  features,offsets = _init_data(a,x)

  T = typeof(features)
  stage1 = Vector{T}(undef,length(r1))
  stage2 = Vector{T}(undef,length(r2))
  for k in eachindex(r1)
    stage1[k] = similar(features)
    stage2[k] = similar(features)
  end

  Δ = length(first(r1))
  
  @inbounds for (j,aj) in enumerate(a.data.bundle)
    ax = lazy_map(aj,x)
    cache = array_cache(ax)
    for i in eachindex(ax)
      axi = getindex!(cache,ax,i)
      for k in eachindex(r1)
        o = offsets[j+(k-1)*Δ]
        _fill_stage!(stage1[k],axi,o,r1[k],i,np)
        _fill_stage!(stage2[k],axi,o,r2[k],i,np)
      end
    end
  end

  stage1,stage2
end

function _fill_stage!(stage,axi,o,trange,i,np)
  for it in enumerate(trange), p in 1:np
    ipt = (it-1)*np+p
    _set_data!(stage,param_getindex(axi,ipt),o,i,p)
  end
end

function _offsets(a::RolloutData{<:FeatureBundle},x::Point)
  ranges = stage_ranges(a)
  Δ = length(first(ranges))
  o = zeros(Int,length(a.bundle))
  for (i,f) in enumerate(a.bundle)
    T = return_type(f,x)
    o[i] = num_components(eltype(T))
  end
  offsets = repeat(o,Δ)
  pushfirst!(offsets,0)
  length_to_ptrs!(offsets)
  return offsets
end