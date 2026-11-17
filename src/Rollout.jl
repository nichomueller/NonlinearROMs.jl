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
  steps = fill(step,length(times)÷step)
  pushfirst!(steps,0)
  length_to_ptrs!(steps)
  last(steps)-1 != length(times) && @warn "Removing $((times[last(steps)],times[end])) from rollout: incomplete cluster"
  RolloutData(data,times[1:steps[end]-1],steps)
end

function RolloutData(data::TransientParamFESolution,step::Int=2)
  times = get_times(data.odesol.r)
  RolloutData(data,times,step)
end

function TwoStageBuckets(a::RolloutData)
  TwoStageBuckets(a,a.steps)
end

ParamDataStructures.num_params(a::RolloutData) = num_params(a.data)

function stage_ranges(a::RolloutData;kwargs...)
  stage_ranges(a.steps;kwargs...)
end

_num_clusters(a::RolloutData) = length(a.steps)-1

function InputSpace(x::AbstractVector{<:Point},a::RolloutData)
  coords = points_to_coords(x,num_params(a)*_num_clusters(a))
  features = _get_features(a,x)
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

function OutputSpace(a::RolloutData{<:TransientParamFESolution})
  r = a.data.odesol.r
  f = a.data.trial(r)
  x = get_coords(f)
  values = _get_values(a,x)
  OutputSpace(f,values)
end

function OutputSpace(sol::TransientParamFESolution,args...)
  times = get_times(sol.odesol.r)
  data = RolloutData(sol,times,args...)
  OutputSpace(data)
end

# utils

function stage_ranges(steps;stage=:first)
  t = ()
  for k in 1:length(steps)-1
    ak = steps[k]
    bk = steps[k+1]
    mk = div(ak + bk,2)
    if stage==:first
      t = (t...,ak:mk-1)
    elseif stage==:second
      t = (t...,mk:bk-1)
    else
      @notimplemented "stage must be :first or :second"
    end
  end
  return t
end

function TwoStageBuckets(a::RolloutData,x::AbstractVector{<:Point})
  TwoStageBuckets(a,a.steps,x)
end

function TwoStageBuckets(data,steps::Union{Tuple,AbstractVector},x::AbstractVector{<:Point})
  r1 = stage_ranges(steps;stage=:first)
  r2 = stage_ranges(steps;stage=:second)
  s1,s2 = split_stages(data,r1,r2,x)
  TwoStageBuckets(s1,s2)
end

function split_stages(a,r1,r2)
  @check length(r1) == length(r2) "r1 and r2 must have the same length"
  _split_stages(a,r1,r2)
end

function split_stages(a,r1,r2,x)
  @check length(r1) == length(r2) "r1 and r2 must have the same length"
  _split_stages(a,r1,r2,x)
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

"""
    _split_stages(a::RolloutData{<:FeatureBundle},r1,r2,x) -> (stage1,stage2)

Per cluster `k`, channel-stacks the bundle's features over the `Δ = length(r1[k])`
global times in `r1[k]` into `stage1[k]` (shape `(D*Δ,N,np)`), and likewise `r2[k]`
into `stage2[k]`. `o[j]` is bundle element `j`'s own channel offset *within one
time slice* (from `_offsets(a.data,x)`); `_fill_stage!` shifts it by `D*(pos-1)`
for the time slice's stage-local position `pos`, and reads bundle element `j`'s
flat (time-major, param-minor) value at global time `t` via `(t-1)*np+p`.
"""
function _split_stages(a::RolloutData{<:FeatureBundle},r1,r2,x::AbstractVector{<:Point})
  @check length(r1) == length(r2) "r1 and r2 must have the same length"
  np = num_params(a.data)
  N = length(x)
  Δ = length(first(r1))

  o = _offsets(a.data,testitem(x))
  D = o[end]-1

  stage1 = [zeros(TData,D*Δ,N,np) for _ in eachindex(r1)]
  stage2 = [zeros(TData,D*Δ,N,np) for _ in eachindex(r2)]

  @inbounds for (j,aj) in enumerate(a.data.bundle)
    oj = o[j]
    ax = lazy_map(aj,x)
    cache = array_cache(ax)
    for i in eachindex(ax)
      axi = getindex!(cache,ax,i)
      for k in eachindex(r1)
        _fill_stage!(stage1[k],axi,oj,D,r1[k],i,np)
        _fill_stage!(stage2[k],axi,oj,D,r2[k],i,np)
      end
    end
  end

  stage1,stage2
end

"""
    _split_stages(a::RolloutData{<:TransientParamFESolution},r1,r2,x) -> (stage1,stage2)

Streams the solution once via `Base.iterate` (never materialising all time
steps at once) and, for every global time `t`, scatters its point-evaluated
state directly into whichever `stage1[k]`/`stage2[k]` slot owns it -- skipping
evaluation for any time beyond the last cluster entirely.
"""
function _split_stages(a::RolloutData{<:TransientParamFESolution},r1,r2,x::AbstractVector{<:Point})
  @check length(r1) == length(r2) "r1 and r2 must have the same length"
  sol = a.data
  np = num_params(sol)
  N = length(x)
  Δ = length(first(r1))

  o = _offsets(sol,testitem(x))
  D = o[end]-1

  stage1 = [zeros(TData,D*Δ,N,np) for _ in eachindex(r1)]
  stage2 = [zeros(TData,D*Δ,N,np) for _ in eachindex(r2)]

  Tmax = last(a.steps)-1
  targets = Vector{Union{Nothing,Tuple{Array{TData,3},Int}}}(nothing,Tmax)
  for k in eachindex(r1)
    for (pos,t) in enumerate(r1[k])
      targets[t] = (stage1[k],pos)
    end
    for (pos,t) in enumerate(r2[k])
      targets[t] = (stage2[k],pos)
    end
  end

  it = iterate(sol)
  t = 1
  while !isnothing(it)
    (_,ah),state = it
    tgt = targets[t]
    if !isnothing(tgt)
      dest,pos = tgt
      ax = lazy_map(ah,x)
      cache = array_cache(ax)
      for i in eachindex(ax)
        axi = getindex!(cache,ax,i)
        _fill_stage!(dest,axi,1,D,pos,i,np)
      end
    end
    t == Tmax && break
    it = iterate(sol,state)
    t += 1
  end

  stage1,stage2
end

# `axi` spans every global time flattened (time-major, param-minor): `trange`
# are the actual global time indices, so `t` (not the stage-local `pos`) picks
# out the flat param index, while `pos` picks out the destination's channel block
function _fill_stage!(stage,axi,o,D,trange::AbstractVector,i,np)
  for (pos,t) in enumerate(trange), p in 1:np
    ipt = (t-1)*np+p
    _set_data!(stage,param_getindex(axi,ipt),o+(pos-1)*D,i,p)
  end
end

# `axi` spans only the current global time's `np` params (one `Base.iterate`
# step of a `TransientParamFESolution`): no flat time index needed, `pos` only
# picks out the destination's channel block
function _fill_stage!(stage,axi,o,D,pos::Int,i,np)
  for p in 1:np
    _set_data!(stage,param_getindex(axi,p),o+(pos-1)*D,i,p)
  end
end

function _get_features(a::RolloutData,x::AbstractVector{<:Point})
  _rollout_data(a,x)
end

function _get_values(a::RolloutData,x::AbstractVector{<:Point})
  _rollout_data(a,x)
end

# un-splits each cluster (stage1 ++ stage2 along the channel axis, restoring
# the full `(D*step,N,np)` cluster) then stacks every cluster along the
# sample axis -- one sample per (original parameter,cluster) pair
function _rollout_data(a::RolloutData,x::AbstractVector{<:Point})
  buckets = TwoStageBuckets(a,x)
  per_cluster = map((s1,s2) -> cat(s1,s2;dims=1),buckets.first_stage,buckets.second_stage)
  cat(per_cluster...;dims=3)
end

function _offsets(a::TransientParamFESolution,x::Point)
  trial = a.trial(a.odesol.r)
  _offsets(zero(trial),x)
end