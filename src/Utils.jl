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

  if epoch == 2 || epoch % log.print_every == 0 || epoch == log.max_epochs
    eta_str = if epoch == 1 && log.max_epochs > 1
      "--:--"
    else
      elapsed_fast = time() - log.t_start_fast
      time_per_epoch = epoch > 1 ? elapsed_fast / (epoch - 1) : 0.0
      format_eta(time_per_epoch * (log.max_epochs - epoch))
    end

    msg = "> Epoch: $(lpad(epoch,5)) \t Loss: $(TData(current_loss)) \t ETA: $eta_str"
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
const TCData = ComplexF32
const AbstractInputData{T} = AbstractArray{T,3}

dimension(data::AbstractInputData) = size(data,1)
num_coords(data::AbstractInputData) = size(data,2)
num_samples(data::AbstractInputData) = size(data,3)

"""
    struct TwoStageBuckets{A<:AbstractInputData}
      first_stage::A
      second_stage::A
      num_clusters::Int
    end

The two stages of a [`RolloutData`](@ref) evaluated on a point set, as two
`(D,N,K)` arrays of equal size: `first_stage`/`second_stage` hold, for every
cluster and parameter, the data at the first/second half of the cluster's times
(the time slots of a stage stacked along the channel axis). The `K` samples are
ordered cluster by cluster, so `num_clusters` blocks of `K÷num_clusters`
parameters each. Both stages share a single normalisation score.
"""
struct TwoStageBuckets{A<:AbstractInputData}
  first_stage::A
  second_stage::A
  num_clusters::Int
  function TwoStageBuckets(first_stage::A,second_stage::A,num_clusters=1) where A<:AbstractInputData
    msg = "First and second stage data must have the same dimensions"
    @check size(first_stage) == size(second_stage) msg
    new{A}(first_stage,second_stage,num_clusters)
  end
end

function Base.copy(data::TwoStageBuckets)
  TwoStageBuckets(
    copy(data.first_stage),
    copy(data.second_stage),
    data.num_clusters
  )
end

dimension(data::TwoStageBuckets) = dimension(data.first_stage)
num_coords(data::TwoStageBuckets) = num_coords(data.first_stage)
num_samples(data::TwoStageBuckets) = num_samples(data.first_stage)
num_clusters(data::TwoStageBuckets) = data.num_clusters

merge_data(data::TwoStageBuckets) = _merge_data(data.first_stage,data.second_stage)

# normalisation handling

abstract type ScoreStyle end

struct MScoreStyle <: ScoreStyle end
struct RScoreStyle <: ScoreStyle end
struct ZScoreStyle <: ScoreStyle end

Score(::ScoreStyle,data;kwargs...) = @abstractmethod
Score(::MScoreStyle,data;kwargs...) = MScore(data;kwargs...)
Score(::RScoreStyle,data;kwargs...) = RScore(data;kwargs...)
Score(::ZScoreStyle,data;kwargs...) = ZScore(data;kwargs...)

# one score shared by both stages: the network's prediction (second-stage scale)
# is fed back as a state, and compared against first-stage targets. The stats are
# computed on the stages stacked along the samples, then both stages are
# normalised in place (merging makes a copy, so normalising it would be lost)
for T in (:MScoreStyle,:RScoreStyle,:ZScoreStyle)
  @eval begin
    function Score(style::$T,data::TwoStageBuckets;normalise=true)
      all_data = cat(data.first_stage,data.second_stage;dims=3)
      score = Score(style,all_data;normalise=false)
      normalise && normalise!(data,score)
      return score
    end
  end
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

for f in (:normalise!,:denormalise!)
  @eval begin
    function $f(data::TwoStageBuckets,score::AbstractScore)
      $f(data.first_stage,score)
      $f(data.second_stage,score)
      data
    end
  end
end
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
  # Compute one scale factor per channel so multi-scale fields are balanced
  s = dropdims(maximum(abs,data;dims=(2,3)),dims=(2,3))
  T = eltype(s)
  @inbounds for i in eachindex(s)
    if s[i] <= floatmin(T)
      s[i] = one(T)
    end
  end
  return MScore(s)
end

normalise!(data::AbstractInputData,score::MScore{<:Number}) = (data ./= score.scale)
denormalise!(data::AbstractInputData,score::MScore{<:Number}) = (data .*= score.scale)

function normalise!(data::AbstractInputData,score::MScore)
  @inbounds @views for (d,vd) in enumerate(eachslice(data,dims=1))
    data[d,:,:] .= vd ./ score.scale[d]
  end
end

function denormalise!(data::AbstractInputData,score::MScore)
  @inbounds @views for (d,vd) in enumerate(eachslice(data,dims=1))
    data[d,:,:] .= vd .* score.scale[d]
  end
end

function combine_scores(score::NoMScore...;kwargs...)
  first(score)
end

function combine_scores(score::MScore{<:Number}...;kwargs...)
  m = zero(first(score).scale)
  for s in score
    m = max(m,abs(s.scale))
  end
  return MScore(m)
end

function combine_scores(score::MScore...;kwargs...)
  D = length(first(score).scale)
  T = eltype(first(score).scale)
  m = zeros(T,D)
  @inbounds for s in score
    for d in 1:D
      m[d] = max(m[d],abs(s.scale[d]))
    end
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
  T = eltype(l)
  # Avoid dividing by zero on constant or near-constant channels without skipping small-scale fields
  @inbounds for i in eachindex(l)
    tol = max(floatmin(T),T(16) * eps(T) * max(abs(l[i]),abs(u[i])))
    if (u[i] - l[i]) <= tol
      l[i] = zero(T)
      u[i] = one(T)
    end
  end
  return RScore(l,u)
end

function normalise!(data::AbstractInputData,score::RScore)
  @inbounds @views for (d,vd) in enumerate(eachslice(data,dims=1))
    data[d,:,:] .= (vd .- score.lower[d]) ./ (score.upper[d] - score.lower[d])
  end
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
  D,N,K = size(data)
  n = N * K
  T = float(eltype(data))
  μ = Vector{T}(undef,D)
  σ = Vector{T}(undef,D)
  @inbounds for d in 1:D
    # Two-pass Float64 accumulation avoids allocating a full Float64 copy of data
    s = 0.0
    for k in 1:K,i in 1:N
      s += Float64(data[d,i,k])
    end
    m = n > 0 ? s / n : 0.0
    ss = 0.0
    for k in 1:K,i in 1:N
      diff = Float64(data[d,i,k]) - m
      ss += diff * diff
    end
    sd = n > 1 ? sqrt(ss / (n - 1)) : 0.0
    tol = max(Float64(floatmin(T)),16.0 * Float64(eps(T)) * abs(m))
    if sd <= tol
      μ[d] = zero(T)
      σ[d] = one(T)
    else
      μ[d] = T(m)
      σ[d] = T(sd)
    end
  end
  return ZScore(μ,σ)
end

function normalise!(data::AbstractInputData,stats::ZScore)
  @inbounds @views for (d,vd) in enumerate(eachslice(data,dims=1))
    data[d,:,:] .= (vd .- stats.μ[d]) ./ stats.σ[d]
  end
end

function denormalise!(data::AbstractInputData,stats::ZScore)
  @inbounds @views for (d,vd) in enumerate(eachslice(data,dims=1))
    data[d,:,:] .= (vd .* stats.σ[d]) .+ stats.μ[d]
  end
end

for S in (NoRScore,NoZScore,NoMScore),f in (:normalise!,:denormalise!)
  @eval begin
    $f(data,score::$S) = data
    $f(data::AbstractInputData,score::$S) = data
    $f(data::TwoStageBuckets,score::$S) = data
  end
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
  if !style.normalise
    # Return no-op scores without mutating input or output tensors
    input_score = Score(style.input_style,map(_ -> nothing,input))
    output_score = Score(style.output_style,nothing)
    return Normalisation(input_score,output_score)
  end
  input_score = Score(style.input_style,input)
  output_score = Score(style.output_style,output)
  Normalisation(input_score,output_score)
end

for f in (:normalise!,:denormalise!)
  @eval begin
    function $f(inputs::NamedTuple,stats::Normalisation)
      $f(inputs,stats.input_score)
    end

    function $f(outputs::Union{AbstractInputData,TwoStageBuckets},stats::Normalisation)
      $f(outputs,stats.output_score)
    end
  end
end

# losses

assemble_norm(args...) = @abstractmethod

"""
    struct EnergyLoss{A<:AbstractMatrix,B<:AbstractVector,C<:AbstractVector} <: Lux.AbstractLossFunction
      X::A
      rows::B
      cols::B
      vals::C
    end

    EnergyLoss(op::AssembleOperator,f)

Sample-averaged squared energy norm `(1/K) Σ_k (ŷ_k - y_k)' X (ŷ_k - y_k)`,
with `X` the Gram matrix of the `GridapROMs` norm `op` (`ℓ2()`, `L2()`, `H1()`,
...) on `f`'s point layout (see [`assemble_norm`](@ref)). `X` is either `(N,N)`,
shared by every channel, or `(D*N,D*N)`, block-diagonal with one block per
channel (the multi-field case). The sparse triplets `(rows,cols,vals)` are
precomputed at construction time (and moved to `XDEV` during training) to avoid
calling `findnz` at every training step.
"""
struct EnergyLoss{A<:AbstractMatrix,B<:AbstractVector,C<:AbstractVector} <: Lux.AbstractLossFunction
  X::A
  rows::B
  cols::B
  vals::C
end

function EnergyLoss(X::AbstractSparseMatrix)
  rows,cols,vals = findnz(X)
  vals_t = eltype(vals) === TData ? vals : TData.(vals)
  EnergyLoss(X,rows,cols,vals_t)
end

EnergyLoss(X::AbstractMatrix) = EnergyLoss(sparse(X))
EnergyLoss(op::AssembleOperator,f) = EnergyLoss(assemble_norm(op,f))

function Lux.unsafe_apply_loss(loss::EnergyLoss,ŷ,y)
  e = ŷ .- y
  D,N,K = size(e)
  w = reshape(loss.vals,1,1,length(loss.vals))
  if size(loss.X,1) == N
    e_p = permutedims(e,(1,3,2))
    gathered = NNlib.gather(e_p,loss.cols)
    Xe_p = NNlib.scatter(+,gathered .* w,loss.rows;dstsize=(D,K,N))
    return sum(e_p .* Xe_p) / TData(K)
  else
    @check size(loss.X,1) == D * N "Norm matrix of size $(size(loss.X)) incompatible with data of size $(size(e))"
    e_p = reshape(permutedims(e,(3,2,1)),1,K,N * D)
    gathered = NNlib.gather(e_p,loss.cols)
    Xe_p = NNlib.scatter(+,gathered .* w,loss.rows;dstsize=(1,K,N * D))
    return sum(e_p .* Xe_p) / TData(K)
  end
end

# visualisation

"""
    visualise(dir,uh,ûh;trian,dir_name="plot",variable_name="uh",param_id=1)

Writes the true field `uh`, the prediction `ûh` and their difference for the
`param_id`-th parameter to `joinpath(dir,dir_name)`, as the cell fields
`variable_name`, `"approx_"*variable_name` and `"err_"*variable_name` (one per
field, suffixed `_field<i>`, for multi-field functions). For a steady `uh` (a
`ParamFEFunction`) a single `.vtu` file is written; for a transient `uh` (a
`TransientParamFESolution`), `ûh` stores every (parameter,time) pair with the
parameter index running fastest, as returned by [`to_element`](@ref) on a rollout
prediction, and one `.vtu` file per time step is written, collected into a `.pvd`
time series.
"""
function visualise(
  dir,uh::ParamFEFunction,ûh::ParamFEFunction;
  trian=get_triangulation(get_fe_space(uh)),
  dir_name="plot",variable_name="uh",param_id=1
  )

  dir = joinpath(dir,dir_name)
  mkpath(dir)
  fields = _get_fields(uh,param_id,ûh,param_id,variable_name)
  writevtk(trian,joinpath(dir,variable_name),cellfields=fields)
end

function visualise(
  dir,uht::TransientParamFESolution,ûh::ParamFEFunction;
  trian=get_triangulation(uht.trial),
  dir_name="plot",variable_name="uh",param_id=1
  )

  r = uht.odesol.r
  np = num_params(r)
  times = get_times(r)
  nt = param_length(ûh) ÷ np
  dir = joinpath(dir,dir_name)
  mkpath(dir)
  createpvd(joinpath(dir,variable_name)) do pvd
    for (it,uh) in enumerate(uht)
      it > nt && break
      fields = _get_fields(uh,param_id,ûh,(it-1)*np+param_id,variable_name)
      file = joinpath(dir,"$(variable_name)_$(it).vtu")
      pvd[times[it]] = createvtk(trian,file,cellfields=fields)
    end
  end
end

function _get_fields(uh,ip,ûh,îp,variable_name)
  fields = Pair{String}[]
  uhip = param_getindex(uh,ip)
  ûhip = param_getindex(ûh,îp)
  _push_fields!(fields,uhip,ûhip,variable_name)
  fields
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