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

abstract type ScoreStyle end

struct MScoreStyle <: ScoreStyle end
struct RScoreStyle <: ScoreStyle end
struct ZScoreStyle <: ScoreStyle end

Score(::ScoreStyle,data;kwargs...) = @abstractmethod
Score(::MScoreStyle,data;kwargs...) = MScore(data;kwargs...)
Score(::RScoreStyle,data;kwargs...) = RScore(data;kwargs...)
Score(::ZScoreStyle,data;kwargs...) = ZScore(data;kwargs...)

function Score(style::NamedTuple,data::NamedTuple;kwargs...)
  scores = NamedTuple()
  for (k,v) in pairs(data)
    scores = merge(scores,(k => Score(style[k],v;kwargs...),))
  end
  return NamedScores(scores)
end

abstract type AbstractScore end

normalise!(data,score::AbstractScore) = @abstractmethod
denormalise!(data,score::AbstractScore) = @abstractmethod

# e.g. the coordinates, when the formatter does not pass them
normalise!(data::Nothing,score::AbstractScore) = data
denormalise!(data::Nothing,score::AbstractScore) = data

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

struct RScore{A} <: AbstractScore
  lower::A
  upper::A
end

const NoRScore = RScore{Nothing}

RScore(::Nothing;kwargs...) = RScore(nothing,nothing)

function RScore(data::AbstractInputData;normalise=true)
  if normalise
    stats = RScore(data;normalise=false)
    normalise!(data,stats)
    return stats
  end
  l = minimum(data,dims=(2,3))
  u = maximum(data,dims=(2,3))
  return RScore(l,u)
end

function normalise!(data::AbstractInputData,score::NoRScore)
  data
end

function normalise!(data::AbstractInputData,score::RScore)
  data .-= score.lower
  data ./= (score.upper - score.lower)
  data
end

function denormalise!(data::AbstractInputData,score::NoRScore)
  data
end

function denormalise!(data::AbstractInputData,score::RScore)
  data .*= (score.upper - score.lower)
  data .+= score.lower
  data
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
  μ = mean(data,dims=(2,3))
  σ = std(data,dims=(2,3))
  # Avoid wiping out or dividing by zero if a channel is constant
  @inbounds for i in eachindex(σ)
    if σ[i] < TData(1e-6)
      μ[i] = zero(eltype(μ))
      σ[i] = one(eltype(σ))
    end
  end
  return ZScore(μ,σ)
end

function normalise!(data::AbstractArray,score::NoZScore)
  data
end

function normalise!(data::AbstractArray,stats::ZScore)
  data .-= stats.μ
  data ./= stats.σ
  data
end

function denormalise!(data::AbstractArray,score::NoZScore)
  data
end

function denormalise!(data::AbstractArray,stats::ZScore)
  data .*= stats.σ
  data .+= stats.μ
  data
end

struct NamedScores{A} <: AbstractScore
  scores::A
end

for f in (:normalise!,:denormalise!)
  @eval begin
    function $f(data::NamedTuple,stats::NamedScores{<:NamedTuple})
      @inbounds for k in keys(stats.scores)
        $f(data[k],stats.scores[k])
      end
      data
    end
  end
end

"""
    struct NormStyle{A<:NamedTuple,B<:ScoreStyle}
      input_style::A
      output_style::B
      normalise::Bool
    end

Recipe for a [`Normalisation`](@ref): which [`ScoreStyle`](@ref) to apply to the
coordinates, the features and the output values, and whether to normalise at all.
Not to be confused with `GridapROMs.NormStyle`, the supertype of the `ℓ2`/`L2`/`H1`
norms used by [`EnergyLoss`](@ref).
"""
struct NormStyle{A<:NamedTuple,B<:ScoreStyle}
  input_style::A
  output_style::B
  normalise::Bool
end

function NormStyle(;
  coords_style=RScoreStyle(),
  features_style=ZScoreStyle(),
  output_style=ZScoreStyle(),
  normalise=true
  )

  input_style = (coords=coords_style,features=features_style)
  NormStyle(input_style,output_style,normalise)
end

struct Normalisation{A<:NamedScores,B<:AbstractScore}
  input_score::A
  output_score::B
end

function Normalisation(style::NormStyle,(input,output);kwargs...)
  normalise = style.normalise
  input_score = Score(style.input_style,input;normalise)
  output_score = Score(style.output_style,output;normalise)
  Normalisation(input_score,output_score)
end

function Normalisation((input,output);kwargs...)
  Normalisation(NormStyle(;kwargs...),(input,output))
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

assemble_norm(args...) = @abstractmethod

"""
    struct EnergyLoss{A<:AbstractMatrix} <: Lux.AbstractLossFunction
      X::A
    end

    EnergyLoss(op::AssembleOperator,f)

MSE between `Xŷ` and `Xy`, with `X` the Gram matrix of the `GridapROMs` norm `op`
(`ℓ2()`, `L2()`, `H1()`, ...) on `f`'s point layout (see [`assemble_norm`](@ref)).
`X` is either `(N,N)`, shared by every channel, or `(D*N,D*N)`, block-diagonal
with one block per channel (the multi-field case).
"""
struct EnergyLoss{A<:AbstractMatrix} <: Lux.AbstractLossFunction
  X::A
end

EnergyLoss(op::AssembleOperator,f) = EnergyLoss(assemble_norm(op,f))

function Lux.unsafe_apply_loss(loss::EnergyLoss,ŷ,y)
  Xŷ = _norm_mul_data(loss.X,ŷ)
  Xy = _norm_mul_data(loss.X,y)
  return Lux.unsafe_apply_loss(Lux.MSELoss(;agg=mean),Xŷ,Xy)
end

function _norm_mul_data(X::AbstractMatrix,x::AbstractInputData)
  D,N,K = size(x)
  if size(X,1) == N
    _mat_mul_data(X,x)
  else
    @check size(X,1) == D*N "Norm matrix of size $(size(X)) incompatible with data of size $(size(x))"
    x_flat = reshape(permutedims(x,(2,1,3)),1,N*D,K)
    y_flat = _mat_mul_data(X,x_flat)
    permutedims(reshape(y_flat,N,D,K),(2,1,3))
  end
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