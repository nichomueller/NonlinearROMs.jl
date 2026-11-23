"""
    struct RolloutData{A,B<:AbstractVector,C<:Union{Tuple,AbstractVector}}
      data::A
      times::B
      steps::C
    end

    RolloutData(data,times,step=2)

Temporal-bundling layout (Brandstetter et al.) of transient `data` (a
[`FeatureBundle`](@ref) of transient `ParamFunction`s, or a
`TransientParamFESolution`) for an autoregressive neural operator. `steps` are
pointers into `times` (indices, not time values) cutting them into `nc` disjoint
clusters of `step` consecutive times, `steps[c]:steps[c+1]-1`; each cluster is
split into two halves (stages) of `step÷2` time slots each. A trailing incomplete
cluster is dropped, with a warning.

Evaluated into an [`InputSpace`](@ref)/[`OutputSpace`](@ref), `RolloutData`
produces [`TwoStageBuckets`](@ref): the time slots of a stage are stacked along
the channel axis (slot-major), and the samples are ordered cluster by cluster,
sample `(c-1)*np+ip` holding cluster `c` of parameter `ip`.
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

num_clusters(a::RolloutData) = length(a.steps)-1

cluster_size(a::RolloutData) = div(length(a.times),num_clusters(a))

stage_size(a::RolloutData) = div(cluster_size(a),2)

# index in a.times of the s-th slot of the k-th stage (k = 1,2) of cluster c
stage_time(a::RolloutData,c::Int,k::Int,s::Int) = a.steps[c] + (k-1)*stage_size(a) + s - 1

num_samples(a::RolloutData) = num_samples(a.data)*num_clusters(a)

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
  # the space only spans the times kept by the rollout (split_stages reads the
  # solution from its first time step, and an incomplete last cluster is dropped)
  r = a.data.odesol.r[:,1:length(a.times)]
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

function split_stages(a::RolloutData{<:FeatureBundle},x::AbstractVector{<:Point})
  np = num_samples(a.data)
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
        r3 = coffsets[c]:coffsets[c+1]-1
        for (k,stage) in enumerate((stage1,stage2)), s in 1:stage_size(a)
          t = stage_time(a,c,k,s)
          o = offsets[(s-1)*(nd+1)+j]
          _fill_stage!(stage,axi,o,i,r3,t,np)
        end
      end
    end
  end

  stage1,stage2
end

function split_stages(a::RolloutData{<:TransientParamFESolution},x::AbstractVector{<:Point})
  nd = 1
  values,offsets = _init_data(a,x)
  coffsets = _sample_offsets(a)

  stage1 = values
  stage2 = similar(values)

  # the solution is only iterable: clusters, stages and slots are visited in
  # time order, i.e. stage_time(a,c,k,s) increases by one at every iteration
  it = iterate(a.data)
  @inbounds for c in 1:num_clusters(a)
    r3 = coffsets[c]:coffsets[c+1]-1
    for stage in (stage1,stage2)
      for s in 1:stage_size(a)
        ah,state = it
        o = offsets[(s-1)*(nd+1)+1]
        ax = lazy_map(ah,x)
        cache = array_cache(ax)
        for i in eachindex(ax)
          axi = getindex!(cache,ax,i)
          _fill_stage!(stage,axi,o,i,r3)
        end
        it = iterate(a.data,state)
      end
    end
  end

  stage1,stage2
end

# axi stores (param,time) pairs with the param index running fastest
function _fill_stage!(stage,axi,o,i,r3,it,np)
  for (ip,k) in enumerate(r3)
    ipt = (it-1)*np+ip
    _set_data!(stage,param_getindex(axi,ipt),o,i,k)
  end
end

function _fill_stage!(stage,axi,o,i,r3)
  for (ip,k) in enumerate(r3)
    _set_data!(stage,param_getindex(axi,ip),o,i,k)
  end
end

for f in (:_get_features,:_get_values)
  @eval begin
    function $f(a::RolloutData,x::AbstractVector{<:Point})
      s1,s2 = split_stages(a,x)
      nc = num_clusters(a)
      TwoStageBuckets(s1,s2,nc)
    end
  end
end

# inverse of split_stages: sample (c,ip) of stage k stores, in channel slot s, the
# state at time t = stage_time(c,k,s); the free values are ordered as the
# transient param arrays, i.e. column (t-1)*np+ip (param index running fastest)
function _free_values(f::FESpace,data::TwoStageBuckets)
  msg = "Data does not match the space's dof count"
  comps = get_free_dof_to_comp(f)
  points = get_free_dof_to_point(f)
  ids = findall(get_node_masks(f)) # representative dof ids
  point_to_pos = Dict(points[dof] => pos for (pos,dof) in enumerate(ids))
  D = maximum(comps)               # components of the field
  ns = dimension(data) ÷ D         # time slots per stage
  @check D*ns == dimension(data) && D*num_coords(data) == length(comps) msg
  nc = num_clusters(data)
  np = num_samples(data) ÷ nc
  T = eltype2(get_vector_type(f))
  u = zeros(T,length(comps),2*ns*nc*np)
  @inbounds @views for dof in eachindex(comps)
    pos = point_to_pos[points[dof]]
    for c in 1:nc, (k,stage) in enumerate((data.first_stage,data.second_stage)), s in 1:ns
      t = (c-1)*2*ns + (k-1)*ns + s
      ch = (s-1)*D + comps[dof]
      u[dof,(t-1)*np+1:t*np] .= stage[ch,pos,(c-1)*np+1:c*np]
    end
  end
  u
end

function _offsets(a::TransientParamFESolution,x::Point)
  trial = a.trial(a.odesol.r)
  _offsets(zero(trial),x)
end

function _offsets(a::RolloutData,x::Point)
  Δ = stage_size(a)
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