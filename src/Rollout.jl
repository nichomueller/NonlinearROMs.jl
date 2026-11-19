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

function TwoStageBuckets(a::RolloutData,x::AbstractVector{<:Point})
  TwoStageBuckets(a,a.steps,x)
end

function stage_ranges(a::RolloutData;kwargs...)
  stage_ranges(a.steps;kwargs...)
end

num_clusters(a::RolloutData) = length(a.steps)-1

cluster_size(a::RolloutData) = div(length(a.times),num_clusters(a))

num_samples(a::RolloutData) = num_samples(a.data)*cluster_size(a)

function InputSpace(x::AbstractVector{<:Point},a::RolloutData)
  coords = points_to_coords(x,num_samples(a))
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

function _split_stages(a::RolloutData{<:FeatureBundle},r1,r2,x::AbstractVector{<:Point})
  np = num_params(a.data)
  nd = length(a.data.bundle)
  features,offsets = _init_data(a,x)
  coffsets = _sample_offsets(a)

  stage1 = features
  stage2 = similar(features)
  
  @inbounds for (j,aj) in enumerate(a.data.bundle)
    ax = lazy_map(aj,x)
    cache = array_cache(ax)
    for i in eachindex(ax)
      axi = getindex!(cache,ax,i)
      for c in 1:num_clusters(a)
        _r3 = coffsets[c]:coffsets[c+1]-1
        for s in 1:cluster_size(a)
          _r1 = offsets[(s-1)*(nd+1)+j]:offsets[(s-1)*(nd+1)+j+1]-1
          _fill_stage!(stage1,axi,o,t1,i,np)
          _fill_stage!(stage2,axi,o,t2,i,np)
        end
      end
      # for k in eachindex(r1)
      #   for (ik,(t1,t2)) in enumerate(zip(r1[k],r2[k]))
      #     o = offsets[(ik-1)*(nd+1)+j]
      #     _fill_stage!(stage1,axi,o,t1,i,np)
      #     _fill_stage!(stage2,axi,o,t2,i,np)
      #   end
      # end
    end
  end

  stage1,stage2
end

function _split_stages(a::RolloutData{<:TransientParamFESolution},r1,r2,x::AbstractVector{<:Point})
  np = num_params(a.data)
  nd = 1
  values,offsets = _init_data(a,x)

  stage1 = values
  stage2 = similar(values)

  ah,state = iterate(a.data)
  for k in eachindex(r1)
    for (stage,rk) in ((stage1,r1[k]),(stage2,r2[k]))
      for ik in eachindex(rk)
        o = offsets[(ik-1)*(nd+1)+j]
        ax = lazy_map(ah,x)
        cache = array_cache(ax)
        for i in eachindex(ax)
          axi = getindex!(cache,ax,i)
          _fill_stage!(stage,axi,o,i,np)
        end
        ah,state = iterate(a.data,state)
      end
    end
  end

  stage1,stage2
end

function _fill_stage!(stage,axi,o,t::Integer,i,np)
  for ip in 1:np
    ipt = (t-1)*np+ip
    _set_data!(stage,param_getindex(axi,ipt),o,i,ip)
  end
end

function _fill_stage!(stage,axi,o,i,np)
  for ip in 1:np
    _set_data!(stage,param_getindex(axi,ip),o,i,ip)
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
  Δ = cluster_size(a)
  o = _offsets(a.data,x)
  ptrs_to_length!(o)
  offsets = repeat(o,Δ)
  length_to_ptrs!(offsets)
  return offsets
end

function _sample_offsets(a::RolloutData)
  offsets = fill(num_samples(a.data),num_clusters(a)+1)
  length_to_ptrs!(offsets)
  return offsets
end