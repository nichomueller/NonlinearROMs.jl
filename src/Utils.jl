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

function ptrs_to_length!(ptrs::AbstractArray{<:Integer})
  @inbounds for i in (length(ptrs)-1):-1:1
    ptrs[i+1] -= ptrs[i]
  end
  ptrs[1] = 0
end

# input data types

const TData = Float32
const AbstractInputData{T} = AbstractArray{T,3}

dimension(data::AbstractInputData) = size(data,1)
num_coords(data::AbstractInputData) = size(data,2)
num_samples(data::AbstractInputData) = size(data,3)

struct TwoStageBuckets{A<:AbstractInputData}
  first_stage::Vector{A}
  second_stage::Vector{A}
end

function TwoStageBuckets(data,steps::Union{Tuple,AbstractVector},args...)
  r1 = stage_ranges(steps;stage=:first)
  r2 = stage_ranges(steps;stage=:second)
  s1,s2 = split_stages(data,r1,r2,args...)
  TwoStageBuckets(s1,s2)
end

function Base.copy(a::TwoStageBuckets)
  TwoStageBuckets(copy(a.first_stage),copy(a.second_stage))
end

dimension(data::TwoStageBuckets) = dimension(first(data.first_stage))
num_coords(data::TwoStageBuckets) = num_coords(first(data.first_stage))
num_samples(data::TwoStageBuckets) = num_samples(first(data.first_stage))

# normalisation handling

abstract type ScoreStyle end

struct MScoreStyle <: ScoreStyle end
struct RScoreStyle <: ScoreStyle end
struct ZScoreStyle <: ScoreStyle end

Score(::ScoreStyle,data;kwargs...) = @abstractmethod
Score(::MScoreStyle,data;kwargs...) = MScore(data;kwargs...)
Score(::RScoreStyle,data;kwargs...) = RScore(data;kwargs...)
Score(::ZScoreStyle,data;kwargs...) = ZScore(data;kwargs...)

function Score(style::ScoreStyle,data::TwoStageBuckets;kwargs...)
  item = first(data.first_stage)
  sample_size = size(item,2)*size(item,3)
  c(args...) = combine_scores(args...;sample_size)
  
  scores1 = map(x -> Score(style,x;kwargs...),data.first_stage) |> c
  scores2 = map(x -> Score(style,x;kwargs...),data.second_stage) |> c
  return c(scores1,scores2)
end

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
combine_scores(scores::AbstractScore...;kwargs...) = @abstractmethod
combine_scores(scores::Union{Tuple,AbstractVector};kwargs...) = combine_scores(scores...;kwargs...)

struct MScore{A} <: AbstractScore
  scale::A
end

const NoMScore = MScore{Nothing}

MScore(::Nothing;kwargs...) = MScore(nothing)

function MScore(data::AbstractInputData;normalise=true)
  if normalise
    stats = MScore(data;normalise=false)
    normalise!(data,stats)
    return stats
  end
  s = maximum(abs,data)
  iszero(s) && (s = one(s))
  return MScore(s)
end

function normalise!(data::AbstractInputData,score::NoMScore)
  data
end

function normalise!(data::AbstractInputData,score::MScore)
  data ./= score.scale
end

function denormalise!(data::AbstractInputData,score::NoMScore)
  data
end

function denormalise!(data::AbstractInputData,score::MScore)
  data .*= score.scale
end

function combine_scores(score::NoMScore...;kwargs...)
  first(score)
end

function combine_scores(score::MScore...;kwargs...)
  m = zero(TData)
  for s in score
    m = max(abs(m),s.scale)
  end
  return MScore(m)
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
  l = dropdims(minimum(data,dims=(2,3)),dims=(2,3))
  u = dropdims(maximum(data,dims=(2,3)),dims=(2,3))
  return RScore(l,u)
end

function normalise!(data::AbstractInputData,score::NoRScore)
  data
end

function normalise!(data::AbstractInputData,score::RScore)
  @inbounds @views for (d,vd) in enumerate(eachslice(data,dims=1))
    data[d,:,:] .= (vd .- score.lower[d]) ./ (score.upper[d] - score.lower[d])
  end
end

function denormalise!(data::AbstractInputData,score::NoRScore)
  data
end

function denormalise!(data::AbstractInputData,score::RScore)
  @inbounds @views for (d,vd) in enumerate(eachslice(data,dims=1))
    data[d,:,:] .= (vd .* (score.upper[d] - score.lower[d])) .+ score.lower[d]
  end
end

function combine_scores(score::NoRScore...;kwargs...)
  first(score)
end

function combine_scores(score::RScore...;kwargs...)
  D = length(first(score).lower)
  l = fill(typemax(TData),D)
  u = fill(typemin(TData),D)
  @inbounds for s in score
    for d in 1:D
      l[d] = min(l[d],s.lower[d])
      u[d] = max(u[d],s.upper[d])
    end
  end
  return RScore(l,u)
end

struct ZScore{A} <: AbstractScore
  μ::A
  σ::A
end

const NoZScore = ZScore{Nothing}

ZScore(::Nothing;kwargs...) = ZScore(nothing,nothing)

function ZScore(data::AbstractInputData;normalise=true)
  if normalise
    stats = ZScore(data;normalise=false)
    normalise!(data,stats)
    return stats
  end
  μ = dropdims(mean(data,dims=(2,3)),dims=(2,3))
  σ = dropdims(std(data,dims=(2,3)),dims=(2,3))
  # A constant channel has zero variance: leave it unnormalised (identity
  # transform) instead of dividing by a near-zero σ, which would collapse it to 0
  @inbounds for i in eachindex(σ)
    if iszero(σ[i])
      μ[i] = zero(eltype(μ))
      σ[i] = one(eltype(σ))
    end
  end
  return ZScore(μ,σ)
end

function normalise!(data::AbstractInputData,score::NoZScore)
  data
end

function normalise!(data::AbstractInputData,stats::ZScore)
  @inbounds @views for (d,vd) in enumerate(eachslice(data,dims=1))
    data[d,:,:] .= (vd .- stats.μ[d]) ./ stats.σ[d]
  end
end

function denormalise!(data::AbstractInputData,score::NoZScore)
  data
end

function denormalise!(data::AbstractInputData,stats::ZScore)
  @inbounds @views for (d,vd) in enumerate(eachslice(data,dims=1))
    data[d,:,:] .= (vd .* stats.σ[d]) .+ stats.μ[d]
  end
end

function Score(style::ZScoreStyle,data::TwoStageBuckets;kwargs...)
  mdata1 = cat(data.first_stage...,dims=3)
  mdata2 = cat(data.second_stage...,dims=3)
  mdata = cat(mdata1,mdata2,dims=3)
  return Score(style,mdata;kwargs...)
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

function Normalisation(style::NormStyle,(input,output))
  input_score = Score(style.input_style,input)
  output_score = Score(style.output_style,output)
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

# agfem compatibility

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

function get_mfdof_to_fdof(f::FESpaceWithLinearConstraints)
  T = eltype(f.mDOF_to_DOF)
  mfdof_to_fdof = zeros(T,num_free_dofs(f))
  for mDOF in 1:length(f.mDOF_to_DOF)
    DOF = f.mDOF_to_DOF[mDOF]
    mdof = FESpaces._DOF_to_dof(mDOF,f.n_fmdofs)
    dof = FESpaces._DOF_to_dof(DOF,f.n_fdofs)
    if mdof > 0
      @check dof > 0
      mfdof_to_fdof[mdof] = dof
    end
  end
  return mfdof_to_fdof
end

for F in (:get_free_dof_to_comp,:get_free_dof_to_point)
  @eval begin
    function $F(f::FESpaceWithLinearConstraints)
      mfdof_to_fdof = get_mfdof_to_fdof(f)
      fdof_to_q = $F(f.space)
      view(fdof_to_q,mfdof_to_fdof)
    end
  end
end

#TODO remove this eventually!

function Algebra.solve(
  fesolver::LinearFESolver,
  feop::ParamFEOperator,
  r::Realisation
  )

  U = get_trial(feop)(r)
  uh = zero(U)
  solve!(uh,fesolver,feop,r)
  FEFunction(U,get_free_dof_values(uh))
end