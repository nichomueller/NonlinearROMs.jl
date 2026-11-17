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

function OutputSpace(uh::RolloutData{<:TransientParamFESolution})
  space = get_fe_space(uh)
  _rollout_output_space(space,uh)
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
  np = num_params(a.data)
  features,offsets = _init_data(a,x)

  T = typeof(features)
  stage1 = Vector{T}(undef,length(r1))
  stage2 = Vector{T}(undef,length(r2))
  @inbounds for k in eachindex(r1)
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
  np = num_params(a.data)
  values,offsets = _init_data(a,x)

  T = typeof(values)
  stage1 = Vector{T}(undef,length(r1))
  stage2 = Vector{T}(undef,length(r2))
  @inbounds for k in eachindex(r1)
    stage1[k] = similar(values)
    stage2[k] = similar(values)
  end

  (r,ah),state = iterate(a.data)
  ax = lazy_map(ah,x)
  cache = array_cache(ax)
  for k in 1:last(a.steps)-1
    for i in eachindex(ax)
      axi = getindex!(cache,ax,i)
      _fill_stage!(stage,axi,o,k,i,np)
    end
    (r,ah),state = iterate(a.data,state)
    ax = lazy_map(ah,x)
  end

  stage1,stage2
end

function _fill_stage!(stage,axi,o,trange::AbstractVector,i,np)
  for it in enumerate(trange), p in 1:np
    ipt = (it-1)*np+p
    _set_data!(stage,param_getindex(axi,ipt),o,i,p)
  end
end

function _fill_stage!(stage,axi,o,it::Int,i,np)
  for p in 1:np
    ipt = (it-1)*np+p
    _set_data!(stage,param_getindex(axi,ipt),o,i,p)
  end
end

function _offsets(a::RolloutData,x::Point)
  ranges = stage_ranges(a)
  Δ = length(first(ranges))
  o = _offsets(a.data,x)
  offsets = repeat(o,Δ)
  pushfirst!(offsets,0)
  length_to_ptrs!(offsets)
  return offsets
end

function _offsets(a::TransientParamFESolution,x::Point)
  trial = a.trial(a.odesol.r)
  _offsets(zero(trial),x)
end