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

Base.copy(a::TwoStageBuckets) = TwoStageBuckets(copy(a.first_stage),copy(a.second_stage))

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

ParamDataStructures.num_params(a::RolloutData) = num_params(a.data)

function stage_ranges(a::RolloutData;kwargs...)
  stage_ranges(a.steps;kwargs...)
end

num_clusters(a::RolloutData) = length(a.steps)-1

function InputSpace(x::AbstractVector{<:Point},a::RolloutData)
  coords = points_to_coords(x,num_params(a)*num_clusters(a))
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

  nel = length(a.data.bundle)
  
  @inbounds for (j,aj) in enumerate(a.data.bundle)
    ax = lazy_map(aj,x)
    cache = array_cache(ax)
    for i in eachindex(ax)
      axi = getindex!(cache,ax,i)
      for k in eachindex(r1)
        for (ik,(t1,t2)) in enumerate(zip(r1[k],r2[k]))
          o = offsets[(ik-1)*(nel+1)+j]
          _fill_stage!(stage1[k],axi,o,t1,i,np)
          _fill_stage!(stage2[k],axi,o,t2,i,np)
        end
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

  it = iterate(a.data)
  for k in eachindex(r1)
    for (stage,rk) in ((stage1[k],r1[k]),(stage2[k],r2[k]))
      for ik in eachindex(rk)
        ah,state = it
        o = offsets[2*ik-1]
        ax = lazy_map(ah,x)
        cache = array_cache(ax)
        for i in eachindex(ax)
          axi = getindex!(cache,ax,i)
          _fill_stage!(stage,axi,o,i,np)
        end
        it = iterate(a.data,state)
      end
    end
  end

  stage1,stage2
end

function _fill_stage!(stage,axi,o,t::Integer,ix,np)
  for ip in 1:np
    ipt = (t-1)*np+ip
    _set_data!(stage,param_getindex(axi,ipt),o,ix,ip)
  end
end

function _fill_stage!(stage,axi,o,ix,np)
  for ip in 1:np
    _set_data!(stage,param_getindex(axi,ip),o,ix,ip)
  end
end

for f in (:_get_features,:_get_values)
  @eval begin
    function $f(a::RolloutData,x::AbstractVector{<:Point})
      TwoStageBuckets(a,x)
    end
  end
end

function _offsets(a::TransientParamFESolution,x::Point)
  trial = a.trial(a.odesol.r)
  _offsets(zero(trial),x)
end

function _offsets(a::RolloutData,x::Point)
  ranges = stage_ranges(a)
  Δ = length(first(ranges))
  o = _offsets(a.data,x)
  ptrs_to_length!(o)
  offsets = repeat(o,Δ)
  length_to_ptrs!(offsets)
  return offsets
end

function sample(s::Sampler,a::TwoStageBuckets)
  TwoStageBuckets(
    map(x -> sample(s,x),a.first_stage),
    map(x -> sample(s,x),a.second_stage)
  )
end

function fetch_data(d::DataFormatter,a::TwoStageBuckets)
  TwoStageBuckets(
    map(x -> fetch_data(d,x),a.first_stage),
    map(x -> fetch_data(d,x),a.second_stage)
  )
end

function _merge_data(a::TwoStageBuckets,b::TwoStageBuckets)
  TwoStageBuckets(
    map(_merge_data,a.first_stage,b.first_stage),
    map(_merge_data,a.second_stage,b.second_stage)
  )
end
