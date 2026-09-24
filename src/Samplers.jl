struct Sampler{A}
  strategy::A
end

Sampler(s::Sampler) = s
Sampler() = Sampler(identity)

sample(args...) = @notimplemented
sample(s::Sampler,x,args...) = @abstractmethod
sample(red::Reduction,x,args...) = sample(get_strategy(red),x,args...)

sample(s::Sampler{typeof(identity)},x,args...) = x

function sample(s::Sampler{<:Function},x::AbstractArray,axis=1)
  y = selectdim(x,axis,1)
  z = s.strategy(y)
  sx = similar(z,(size(z)...,size(x,axis)))
  @views for i in 1:size(x,axis)
    y = selectdim(x,axis,i)
    z = s.strategy(y)
    selectdim(sx,axis,i) .= z
  end
  return sx
end

function sample(s::Sampler{<:Integer},x::AbstractArray,axis=1)
  step = s.strategy
  selectdim(x,axis,1:step:size(x,axis))
end

function sample(s::Sampler{<:Function},x::Realisation,args...)
  map(x) do μ
    s.strategy(μ)
  end |> Realisation
end

function sample(s::Sampler{<:Integer},x::Realisation,args...)
  step = s.strategy
  x[1:step:num_params(x)]
end

function sample(s::Sampler,x::TransientRealisation,args...)
  p = sample(s,get_params(x),args...)
  TransientRealisation(p,get_times(x),get_initial_time(x))
end

function sample(s::Sampler,x::BlockSnapshots,axis=1)
  @notimplemented "Do this!"
end

# coordinate snapshots sampling

for T in (:(typeof(identity)),:Function,:Integer)
  @eval begin
    function sample(s::Sampler{<:$T},x::CoordinateSnapshots,axis=1)
      data = sample(s,x.snaps,axis)
      # ConsecutiveParamVector convention: a (dofs,params*times) matrix, params
      # varying fastest. `data` is (dofs,params) for steady snapshots (already
      # correct) or (dofs,params,times) for transient ones; merging the trailing
      # axes via `reshape` produces exactly that column order either way.
      pdata = ConsecutiveParamArray(reshape(data,size(data,1),:))
      sx = Snapshots(pdata,get_realisation(x))
      xx = sample(s,get_coordinates(x))
      CoordinateSnapshots(sx,xx)
    end
  end
end

struct MultiSampler{A,B,C}
  space_sampler::Sampler{A}
  param_sampler::Sampler{B}
  time_sampler::Sampler{C}
end

function MultiSampler(s,p,t)
  space_sampler = Sampler(s)
  param_sampler = Sampler(p)
  time_sampler = Sampler(t)
  MultiSampler(space_sampler,param_sampler,time_sampler)
end

function MultiSampler(;space_step=1,param_step=1,time_step=nothing)
  MultiSampler(space_step,param_step,time_step)
end

function sample(s::MultiSampler,x::AbstractArray{<:Point})
  sample(s.space_sampler,vec(x))
end

# At inference time only the parameter transform applies (never space/time
# subsampling, since predictions are always required at full resolution).
function sample(s::MultiSampler,x::AbstractRealisation)
  sample(s.param_sampler,x)
end

function param_sample(s::Sampler{typeof(identity)},x::Snapshots)
  x
end

function param_sample(s::Sampler{<:Function},x::Snapshots)
  r = sample(s,get_realisation(x))
  Snapshots(get_param_data(x),get_dof_map(x),r)
end

function param_sample(s::Sampler,x::Snapshots)
  select_snapshots(x,get_param_ids(s,x))
end

function param_sample(s::MultiSampler,x::Snapshots) 
  param_sample(s.param_sampler,x)
end

function time_sample(s::Sampler{typeof(identity)},x::TransientSnapshots) 
  x
end

function time_sample(s::Sampler,x::TransientSnapshots)
  select_times(x,get_time_ids(s,x))
end

function time_sample(s::MultiSampler,x::TransientSnapshots) 
  time_sample(s.time_sampler,x)
end

const SteadyNeuralSampler{A,B} = MultiSampler{A,B,Nothing}

function sample(s::SteadyNeuralSampler,x::SteadySnapshots)
  sp = param_sample(s,x)
  space_axis = 1
  sample(s.space_sampler,sp,space_axis)
end

function sample(s::SteadyNeuralSampler,x::SteadyCoordinateSnapshots)
  sp = param_sample(s,x.snaps)
  sample(s.space_sampler,CoordinateSnapshots(sp,x.coords))
end

const TransientNeuralSampler{A,B,C} = MultiSampler{A,B,C}

function sample(s::TransientNeuralSampler,x::TransientSnapshots)
  xp = param_sample(s,x)
  xpt = time_sample(s,xp)
  space_axis = 1
  sample(s.space_sampler,xpt,space_axis)
end

function sample(s::TransientNeuralSampler,x::TransientCoordinateSnapshots)
  xp = param_sample(s,x.snaps)
  xpt = time_sample(s,xp)
  sample(s.space_sampler,CoordinateSnapshots(xpt,x.coords))
end

get_space_ids(s::MultiSampler,args...) = get_ids(s.space_sampler,args...)
get_param_ids(s::MultiSampler,args...) = get_ids(s.param_sampler,args...)
get_time_ids(s::MultiSampler,args...) = get_ids(s.time_sampler,args...)

# utils

"""
    get_ids(s::Sampler,n::Int) -> AbstractVector{Int}

Resolves the explicit set of indices (out of `1:n`) selected by `s`: the full
range strided by `s.strategy` when it is an `Integer`, `s.strategy` itself when
it is already an `AbstractVector` of indices, or `1:n` (no subsampling) when
`s.strategy` is `nothing`.
"""
get_ids(s::Sampler{<:AbstractVector},n::Int) = s.strategy
get_ids(s::Sampler{<:Integer},n::Int) = 1:s.strategy:n
get_ids(s::Sampler{Nothing},n::Int) = 1:n

for (f,g) in zip((:get_param_ids,:get_time_ids),(:num_params,:num_times))
  @eval begin
    function $f(s::Sampler{<:AbstractVector},x::Snapshots)
      return s.strategy
    end

    function $f(s::Sampler{<:Integer},x::Snapshots)
      step = s.strategy
      ids = 1:step:$g(x)
      return ids
    end

    function $f(s::Sampler{Nothing},x::Snapshots)
      return nothing
    end
  end
end