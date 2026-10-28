# loggers 

mutable struct TrainingLog
  name::String
  max_epochs::Int
  print_every::Int
  verbose::Bool
  t_start::Float64
  t_start_fast::Float64
end

function TrainingLog(name::String,max_epochs::Int;verbose::Bool=true,print_every=500)
  TrainingLog(name,max_epochs,print_every,verbose,0.0,0.0)
end

function init!(log::TrainingLog)
  !log.verbose && return nothing

  log.t_start = time()
  @info "Starting $(log.name) Training on Reactant Device (First epoch compiles XLA...)"
  return nothing
end

function update!(log::TrainingLog,epoch::Int,current_loss::Real)
  !log.verbose && return nothing

  if epoch == 1
    log.t_start_fast = time()
    comp_mins = round((log.t_start_fast - log.t_start) / 60,digits=2)
    @info "Compilation finished in $comp_mins min. Fast training started."
  end

  if epoch == 1 || epoch % log.print_every == 0 || epoch == log.max_epochs
    elapsed_fast = time() - log.t_start_fast
    time_per_epoch = epoch > 1 ? elapsed_fast / (epoch - 1) : 0.0
    eta_seconds = time_per_epoch * (log.max_epochs - epoch)

    msg = "> Epoch: $(lpad(epoch,5)) \t Loss: $(TData(current_loss)) \t ETA: $(format_eta(eta_seconds))"
    println(msg)
  end
  return nothing
end

function finalise!(log::TrainingLog)
  !log.verbose && return nothing

  total_mins = round((time() - log.t_start) / 60,digits=2)
  @info "Training $(log.name) Completed in $total_mins minutes"
  return nothing
end

function format_eta(eta_seconds::Real)
  eta_sec = round(Int,eta_seconds)
  h = div(eta_sec,3600)
  m = div(rem(eta_sec,3600),60)
  s = rem(eta_sec,60)
  if h > 0
    "$(lpad(h,2,'0')):$(lpad(m,2,'0')):$(lpad(s,2,'0'))"
  else
    "$(lpad(m,2,'0')):$(lpad(s,2,'0'))"
  end  
end

# data utilities

function tuple_of_arrays(a)
  function first_and_tail(a)
    x = map(first,a)
    y = map(Base.tail,a)
    x,y
  end

  function take(a,::Type{Tuple{T}} where T)
    x = map(first,a)
    (x,)
  end

  function take(a,::Type{Tuple{A,B}} where {A,B})
    x,y = first_and_tail(a)
    t1, = tuple_of_arrays(y)
    (x,t1)
  end

  function take(a,::Type{Tuple{A,B,C}} where {A,B,C})
    x,y = first_and_tail(a)
    t1,t2 = tuple_of_arrays(y)
    (x,t1,t2)
  end
  
  function take(a,::Type)
    x,y = first_and_tail(a)
    (x,tuple_of_arrays(y)...)
  end

  take(a,eltype(a))
end

# input data type 

const TData = Float32
const AbstractInputData{T} = AbstractArray{T,3}

dimension(data::AbstractInputData) = size(data,1)
num_coords(data::AbstractInputData) = size(data,2)
num_samples(data::AbstractInputData) = size(data,3)

# normalisation handling

abstract type AbstractScore end

normalise!(data,score::AbstractScore) = @abstractmethod
denormalise!(data,score::AbstractScore) = @abstractmethod

struct MScore{A<:Union{Nothing,Number}} <: AbstractScore
  max::A
end

const NoMScore = MScore{Nothing}

MScore(::Nothing;kwargs...) = MScore(nothing)

function MScore(data::AbstractInputData;normalise=true)
  max = maximum(abs,data)
  normalise && (data ./= max)
  MScore(max)
end

function normalise!(data,score::NoMScore)
  data
end

function normalise!(data,score::MScore)
  rdiv!(data,score.max)
end

function denormalise!(data,score::NoMScore)
  data
end

function denormalise!(data,score::MScore)
  rmul!(data,score.max)
end

struct ZScore{A,B} <: AbstractScore
  μ::A
  σ::B
end

const NoZScore = ZScore{Nothing,Nothing}

ZScore(::Nothing;kwargs...) = ZScore(nothing,nothing)

function ZScore(data::AbstractInputData;normalise=true)
  if normalise
    stats = ZScore(data;normalise=false)
    normalise!(data,stats)
    return stats
  end
  μ = dropdims(mean(data,dims=3),dims=3)
  σ = dropdims(std(data,dims=3),dims=3)
  # Avoid dividing by zero if a feature is constant
  @inbounds for i in eachindex(σ)
    iszero(σ[i]) && (σ[i] = one(eltype(σ)))
  end
  return ZScore(μ,σ)
end

function normalise!(data::AbstractMatrix,score::NoZScore)
  data
end

function normalise!(data::AbstractMatrix,stats::ZScore)
  data .-= stats.μ
  data ./= stats.σ
  data
end

function denormalise!(data::AbstractMatrix,score::NoZScore)
  data
end

function denormalise!(data::AbstractMatrix,stats::ZScore)
  data .*= stats.σ
  data .+= stats.μ
  data
end

for f in (:normalise!,:denormalise!)
  @eval begin
    function $f(data::AbstractInputData,stats::ZScore)
      @inbounds for m in eachslice(data,dims=3)
        $f(m,stats)
      end
      data
    end
  end
end

struct Scores{A} <: AbstractScore
  scores::A
end

for f in (:MScore,:ZScore)
  @eval begin
    function $f(data::NamedTuple;kwargs...)
      scores = NamedTuple()
      for (k,v) in pairs(data)
        scores = merge(scores,NamedTuple{(k,)}(($f(v;kwargs...),)))
      end
      return Scores(scores)
    end
  end
end

for f in (:normalise!,:denormalise!)
  @eval begin
    function $f(data::NamedTuple,stats::Scores{<:NamedTuple})
      @inbounds for k in keys(stats.scores)
        $f(data[k],stats.scores[k])
      end
      data
    end
  end
end

struct Normalisation{A<:AbstractScore,B<:AbstractScore}
  input_score::A
  output_score::B
end

function Normalisation((input,output);kwargs...)
  input_score = ZScore(input;kwargs...)
  output_score = MScore(output;kwargs...)
  Normalisation(input_score,output_score)
end

for f in (:normalise!,:denormalise!)
  @eval begin
    function $f(inputs::NamedTuple,stats::Normalisation)
      $f(inputs,stats.input_score)
    end

    function $f(outputs::AbstractInputData,stats::Normalisation)
      $f(outputs,stats.output_score)
    end
  end
end

# losses

assemble_mass(args...) = @abstractmethod
assemble_stiffness(args...) = @abstractmethod
assemble_mass_stiffness(args...) = @abstractmethod

L2Loss(args...) = EnergyLoss(assemble_mass(args...))
H1Loss(args...) = EnergyLoss(assemble_stiffness(args...))
H1L2Loss(args...) = EnergyLoss(assemble_mass_stiffness(args...))

struct EnergyLoss{A<:AbstractMatrix} <: Lux.AbstractLossFunction
  X::A
end

function Lux.unsafe_apply_loss(loss::EnergyLoss,ŷ,y)
  Xŷ = _mat_mul_data(loss.X,ŷ)
  Xy = _mat_mul_data(loss.X,y)
  return Lux.unsafe_apply_loss(Lux.MSELoss(;agg=mean),Xŷ,Xy)
end

# Gridap fixes

function FESpaces.gather_free_and_dirichlet_values!(
  fmdof_to_val::AbstractVector{T},
  dmdof_to_val::AbstractVector{T},
  f::FESpaceWithLinearConstraints,
  cell_to_ludof_to_val
  ) where T

  fdof_to_val = Vector{T}(undef,num_free_dofs(f.space))
  ddof_to_val = Vector{T}(undef,num_dirichlet_dofs(f.space))
  gather_free_and_dirichlet_values!(fdof_to_val,ddof_to_val,f.space,cell_to_ludof_to_val)
  FESpaces._setup_mdof_to_val!(
    fmdof_to_val,
    dmdof_to_val,
    fdof_to_val,
    ddof_to_val,
    f.mDOF_to_DOF,
    f.n_fdofs,
    f.n_fmdofs)
  fmdof_to_val,dmdof_to_val
end

# visualisation

function visualise(
  dir,uh,ûh;
  trian=get_triangulation(get_fe_space(uh)),
  dir_name="plot",variable_name="uh",
  )

  dir = joinpath(dir,dir_name)
  mkpath(dir)
  fields = Pair{String}[]
  for ip in param_eachindex(uh)
    uhip = param_getindex(uh,ip)
    ûhip = param_getindex(ûh,ip)
    _push_fields!(fields,uhip,ûhip,"$(variable_name)_param_$(ip)")
  end
  writevtk(trian,joinpath(dir,variable_name),cellfields=fields)
end

function _push_fields!(fields,uhip,ûhip,name)
  ehip = uhip - ûhip
  push!(fields,name => uhip)
  push!(fields,"approx_"*name => ûhip)
  push!(fields,"err_"*name => ehip)
end

function _push_fields!(fields,uhip::MultiFieldFEFunction,ûhip::MultiFieldFEFunction,name)
  for i in 1:num_fields(uhip)
    _push_fields!(fields,uhip[i],ûhip[i],"$(name)_field$(i)")
  end
end