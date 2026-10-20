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

    msg = "> Epoch: $(lpad(epoch,5)) \t Loss: $(Float32(current_loss)) \t ETA: $(format_eta(eta_seconds))"
    println(msg)
  end
  return nothing
end

function finalize!(log::TrainingLog)
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
  return h > 0 ? "$(lpad(h,2,'0')):$(lpad(m,2,'0')):$(lpad(s,2,'0'))" :
         "$(lpad(m,2,'0')):$(lpad(s,2,'0'))"
end

# normalisation handling 

struct ZScore{A<:AbstractVector,B<:AbstractVector}
  μ::A
  σ::B
end

function ZScore(data::AbstractMatrix;normalise=false)
  if normalise
    stats = ZScore(data;normalise=false)
    normalise!(data,stats)
    return stats
  end
  μ = dropdims(mean(data,dims=2),dims=2)
  σ = dropdims(std(data,dims=2),dims=2)
  # Avoid dividing by zero if a feature is constant
  for i in eachindex(σ)
    iszero(σ[i]) && (σ[i] = one(eltype(σ)))
  end
  return ZScore(μ,σ)
end

struct Normalisation{T<:Real,A<:ZScore,B<:ZScore}
  dmax::T
  pscore::A 
  xscore::B
end

function Normalisation(data,params,coords;normalise=false)
  dmax = maximum(abs,data)
  normalise && (data ./= dmax)
  input = ZScore(params;normalise)
  output = ZScore(coords;normalise)
  Normalisation(dmax,input,output)
end

normalise!(args...) = @abstractmethod
normalise!(data,::typeof(identity)) = data

function normalise!(data::AbstractVector,stats::ZScore)
  data .-= stats.μ
  data ./= stats.σ
  data
end

function normalise!(data::AbstractMatrix,stats::ZScore)
  @inbounds for v in eachcol(data)
    normalise!(v,stats)
  end
  data
end

function normalise!(inout::NTuple{2,AbstractArray},stats::Normalisation)
  a,b = inout
  normalise!(a,stats.pscore)
  normalise!(b,stats.xscore)
end

function normalise!(inout::NTuple{3,AbstractArray},stats::Normalisation)
  a,b,c = inout
  a ./= stats.dmax
  normalise!(b,stats.pscore)
  normalise!(c,stats.xscore)
end

denormalise!(args...) = @abstractmethod
denormalise!(data,::typeof(identity)) = data

function denormalise!(data::AbstractArray,stats::Normalisation)
  data .*= stats.dmax
  data
end

# utils

dimension(μ::Realisation) = length(first(μ))
dimension(μ::TransientRealisation) = dimension(get_params(μ))

function matrix_of_coords(coords::AbstractVector{Point{D,T}}) where {D,T}
  coords_mat = zeros(T,D,length(coords))
  for (i,coord) in enumerate(coords)
    for d in 1:D 
      coords_mat[d,i] = coord.data[d]
    end
  end
  return coords_mat
end

function matrix_of_coords(coords::AbstractVector{Point{D,T}},times::AbstractVector{S}) where {D,T,S}
  TS = promote_type(T,S)
  coords_mat = zeros(TS,D+1,length(coords)*length(times))
  col = 0
  for t in times,coord in coords
    col += 1
    for d in 1:D
      coords_mat[d,col] = coord.data[d]
    end
    coords_mat[D+1,col] = t
  end
  return coords_mat
end