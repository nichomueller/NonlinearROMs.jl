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

rescale!(args...) = @abstractmethod
rescale!(data,::typeof(identity)) = data

function rescale!(data::AbstractArray,stats::Normalisation)
  data .*= stats.dmax
  data
end

# Gridap

function FESpaces.get_free_dof_coordinates(V::MultiFieldFESpace)
  map(get_free_dof_coordinates,V.spaces)
end

# Data types

struct FeaturesBundle{A<:Tuple}
  bundle::A
end

FeaturesBundle(args::Function...) = FeaturesBundle(args)

function get_features(f::FeaturesBundle,op::ParamOperator,r::Realisation)
  trial = get_trial(op)(r)
  features = zeros(num_free_dofs(trial),length(f.bundle),num_params(r))
  @inbounds @views for (j,f) in enumerate(f.bundle)
    fh = interpolate(f,trial)
    fv = get_free_dof_values(fh)
    for k in param_eachindex(fv)
      features[:,j,k] = param_getindex(fv,k)
    end
  end
  return features
end

# utils

function get_formatted_data(::Type{T},values,times) where T
  d = T.(matrix_of_values(values))
  nx = size(d,1)
  nt = length(times)
  np = Int(size(d,2)/nt)
  reshape(permutedims(reshape(d,nx,np,nt),(1,3,2)),:,np)
end

dimension(μ::Realisation) = length(first(μ))
dimension(μ::TransientRealisation) = dimension(get_params(μ))

matrix_of_values(a::AbstractMatrix) = a
matrix_of_values(a::AbstractArray) = reshape(a,size(a,1),:)
matrix_of_values(x::AbstractParamArray) = matrix_of_values(get_all_data(x))

function matrix_of_params(r::AbstractRealisation)
  params = zeros(dimension(r),num_params(r))
  matrix_of_params!(params,r)
end

function matrix_of_params!(params,r::AbstractRealisation)
  @check size(params,2) == num_params(r)
  μ = get_params(r)
  @inbounds @views for i in axes(params,2)
    params[:,i] = μ.params[i]
  end
  params
end

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

#TODO @Isaia: your old tensor_of_coords function stacked params before the coords 
# on the rows, are you sure it's correct? I am doing the opposite here, please fix it 
# in case it's wrong.
function tensor_of_coords(coords::AbstractMatrix{T},params::AbstractMatrix{S}) where {T,S}
  TS = promote_type(T,S)
  D,nx = size(coords)
  P,np = size(params)
  tensor = zeros(TS,D+P,nx,np)
  for (i,x) in enumerate(eachcol(coords))
    for (j,μ) in enumerate(eachcol(params))
      for d in 1:D
        tensor[d,i,j] = x[d]
      end
      for p in 1:P
        tensor[D+p,i,j] = μ[p]
      end
    end
  end
  return tensor
end