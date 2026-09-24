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

struct ZscoreStats{A<:AbstractVector,B<:AbstractVector}
  μ::A
  σ::B
end

function ZscoreStats(data::AbstractMatrix;normalise=false)
  if normalise
    stats = ZscoreStats(data;normalise=false)
    normalise!(data,stats)
    return stats
  end
  μ = dropdims(mean(data,dims=2),dims=2)
  σ = dropdims(std(data,dims=2),dims=2)
  # Avoid dividing by zero if a feature is constant
  for i in eachindex(σ)
    iszero(σ[i]) && (σ[i] = one(eltype(σ)))
  end
  return ZscoreStats(μ,σ)
end

struct NormStats{T<:Real,A<:ZscoreStats,B<:ZscoreStats}
  dmax::T
  pscore::A 
  xscore::B
end

function NormStats(data,params,coords;normalise=false)
  dmax = maximum(abs,data)
  normalise && (data ./= dmax)
  input = ZscoreStats(params;normalise)
  output = ZscoreStats(coords;normalise)
  NormStats(dmax,input,output)
end

normalise!(args...) = @abstractmethod
normalise!(data,::typeof(identity)) = data

function normalise!(data::AbstractVector,stats::ZscoreStats)
  data .-= stats.μ
  data ./= stats.σ
  data
end

function normalise!(data::AbstractMatrix,stats::ZscoreStats)
  @inbounds for v in eachcol(data)
    normalise!(v,stats)
  end
  data
end

function normalise!(inout::NTuple{2,AbstractArray},stats::NormStats)
  a,b = inout
  normalise!(a,stats.pscore)
  normalise!(b,stats.xscore)
end

function normalise!(inout::NTuple{3,AbstractArray},stats::NormStats)
  a,b,c = inout
  a ./= stats.dmax
  normalise!(b,stats.pscore)
  normalise!(c,stats.xscore)
end

rescale!(args...) = @abstractmethod
rescale!(data,::typeof(identity)) = data

function rescale!(data::AbstractArray,stats::NormStats)
  data .*= stats.dmax
  data
end

# Data types

function FESpaces.get_free_dof_coordinates(V::MultiFieldFESpace)
  map(get_free_dof_coordinates,V.spaces)
end

struct CoordinateSnapshots{T,N,Tc,Nc,A<:AbstractSnapshots{T,N},B<:AbstractArray{Tc,Nc}} <: AbstractSnapshots{T,N}
  snaps::A
  coords::B
end

function CoordinateSnapshots(snaps::AbstractSnapshots,V::FESpace)
  coords = get_free_dof_coordinates(V)
  CoordinateSnapshots(snaps,coords)
end

const SteadyCoordinateSnapshots{T,N,Tc,Nc} = CoordinateSnapshots{T,N,Tc,Nc,<:SteadySnapshots{T,N}}
const TransientCoordinateSnapshots{T,N,Tc,Nc} = CoordinateSnapshots{T,N,Tc,Nc,<:TransientSnapshots{T,N}}

ParamDataStructures.get_all_data(s::CoordinateSnapshots) = get_all_data(s.snaps)
ParamDataStructures.get_param_data(s::CoordinateSnapshots) = get_param_data(s.snaps)
ParamDataStructures.get_initial_param_data(s::CoordinateSnapshots) = get_initial_param_data(s.snaps)
DofMaps.get_dof_map(s::CoordinateSnapshots) = get_dof_map(s.snaps)
ParamDataStructures.get_realisation(s::CoordinateSnapshots) = get_realisation(s.snaps)
get_coordinates(s::CoordinateSnapshots) = s.coords

function ParamDataStructures.select_snapshots(s::CoordinateSnapshots,pindex) 
  snaps = select_snapshots(s.snaps,pindex)
  CoordinateSnapshots(snaps,s.coords)
end

function ParamDataStructures.select_times(s::CoordinateSnapshots,tindex) 
  snaps = select_times(s.snaps,tindex)
  CoordinateSnapshots(snaps,s.coords)
end

function Base.getindex(s::CoordinateSnapshots{T,N},i::Vararg{Integer,N}) where {T,N}
  getindex(s.snaps,i...)
end

function Base.setindex!(s::CoordinateSnapshots{T,N},v,i::Vararg{Integer,N}) where {T,N}
  setindex!(s.snaps,v,i...)
end

function get_formatted_data(::Type{T},s::AbstractSnapshots) where T
  data = T.(get_all_data(s))
  params = T.(matrix_of_params(get_realisation(s)))
  return (data,params)
end

function get_formatted_data(::Type{T},s::CoordinateSnapshots) where T
  data,params = get_formatted_data(T,s.snaps)
  coords = T.(matrix_of_coords(get_coordinates(s)))
  return (data,params,coords)
end

function get_formatted_data(::Type{T},s::TransientCoordinateSnapshots) where T
  data_3d,params = get_formatted_data(T,s.snaps) # data_3d: (N_dofs,n_samples,N_time)
  times = get_times(get_realisation(s))
  coords = T.(matrix_of_coords(get_coordinates(s),times)) # (D_phys,N_dofs*N_time)

  N_dofs,n_samples,N_time = size(data_3d)
  data = zeros(T,N_dofs*N_time,n_samples)
  for i in 1:n_samples
    col = 1
    for t_idx in 1:N_time,x_idx in 1:N_dofs
      data[col,i] = data_3d[x_idx,i,t_idx]
      col += 1
    end
  end

  return data,params,coords
end

function get_formatted_data(::Type{T},r::AbstractRealisation,x::AbstractArray{<:Point}) where T
  params = T.(matrix_of_params(r))
  coords = T.(matrix_of_coords(x))
  return (params,coords)
end

function get_formatted_data(::Type{T},r::TransientRealisation,x::AbstractArray{<:Point}) where T
  params = T.(matrix_of_params(r))
  times = get_times(get_realisation(r))
  coords = T.(matrix_of_coords(x,times))
  return (params,coords)
end

function get_formatted_data(s)
  get_formatted_data(Float32,s)
end

# utils 

get_dof_to_nodes(b) = @abstractmethod
get_dof_to_nodes(b::LagrangianDofBasis) = b.nodes[b.dof_to_node]

# utils

dimension(μ::Realisation) = length(first(μ))
dimension(μ::TransientRealisation) = dimension(get_params(μ))

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

_get_data(a) = get_all_data(a)
_get_data(a::AbstractParamMatrix) = reshape(get_all_data(a),innerlength(a),:)
_get_data(a::AbstractMatrix) = a
_get_data(a::AbstractArray{T,3}) where T = reshape(a,:,size(a,3))