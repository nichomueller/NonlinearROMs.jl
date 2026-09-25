const NNHRProjection{A<:NNHyperReduction,B<:Projection} = HRProjection{A,B}

function FESpaces.interpolate!(
  b̂::AbstractArray,
  cache,
  a::NNHRProjection,
  r::AbstractRealisation
  )

  o = one(eltype2(b̂))
  x = matrix_of_params(r)
  i = get_interpolation(a)
  _coeff = i.interpolation(i.interpolation,x)
  coeff = ConsecutiveParamArray(_coeff)
  mul!(b̂,a,coeff,o,o)
  return b̂
end

struct NNRegressor{A,B} <: Map
  model::A
  bias::B
end

function NNRegressor(model::NeuralModel,test::RBSpace)
  T = get_dof_value_type(test)
  nrows = num_reduced_dofs(test)
  basis = ReducedProjection(zeros(T,nrows,1))
  NNRegressor(model,basis)
end

function NNRegressor(model::NeuralModel,trial::RBSpace,test::RBSpace)
  T = get_dof_value_type(trial)
  nrows = num_reduced_dofs(test)
  ncols = num_reduced_dofs(trial)
  basis = ReducedProjection(zeros(T,nrows,ncols,1))
  NNRegressor(model,basis)
end

RBSteady.get_basis(a::NNRegressor) = a.bias
RBSteady.get_style(a::NNRegressor) = NNRegression()
RBSteady.get_interpolation(a::NNRegressor) = EmptyInterpolation()
RBSteady.projection_eltype(a::NNRegressor) = RBSteady.projection_eltype(a.bias)

function FESpaces.interpolate!(
  b̂::AbstractArray,
  cache,
  a::NNRegressor,
  r::AbstractRealisation
  )

  o = one(eltype2(b̂))
  x = matrix_of_params(r)
  i = get_interpolation(a)
  b̂r = i.interpolation(i.interpolation,x)
  _axpy!(o,b̂r,b̂)
  return b̂
end

function RBSteady.HRProjection(
  red::NNRegression,
  s::Snapshots,
  trian::Triangulation,
  test::RBSpace
  )

  r = get_realisation(s)
  b = GalerkinProjectable(s)
  y = galerkin_projection(test,b)
  ϕ = get_basis(y)
  red = get_strategy(red)
  model = train_neural_coefficient(red,r,ϕ)
  return NNRegressor(model,test)
end

function RBSteady.HRProjection(
  red::NNRegression,
  s::Snapshots,
  trian::Triangulation,
  trial::RBSpace,
  test::RBSpace
  )

  r = get_realisation(s)
  A = GalerkinProjectable(s)
  y = galerkin_projection(test,A,trial)
  ϕ = get_basis(y)
  red = get_strategy(red)
  model = train_neural_coefficient(red,r,ϕ)
  return NNRegressor(model,trial,test)
end

function RBSteady.HRProjection(
  red::NNHyperReduction,
  s::Snapshots,
  trian::Triangulation,
  test::RBSpace
  )

  basis = projection(get_reduction(red),s)
  proj_basis = project(test,basis)
  interp = Interpolation(red,basis,s)
  return HRProjection(proj_basis,red,interp)
end

function RBSteady.HRProjection(
  red::NNHyperReduction,
  s::Snapshots,
  trian::Triangulation,
  trial::RBSpace,
  test::RBSpace
  )

  basis = projection(get_reduction(red),s)
  proj_basis = project(test,basis,trial)
  interp = Interpolation(red,basis,s)
  return HRProjection(proj_basis,red,interp)
end

function RBSteady.allocate_coefficient(a::NNRegressor,r::AbstractRealisation)
  x = matrix_of_params(r)
  return_cache(a.model,x)
end

function RBSteady.allocate_coefficient(a::NNHRProjection,r::AbstractRealisation)
  x = matrix_of_params(r)
  i = get_interpolation(a)
  return_cache(i.interpolation,x)
end

"""
"""
const NNContribution = AffineContribution{<:NNHRProjection}

function RBSteady.allocate_coefficient(a::NNContribution,args...)
  allocate_coefficient(first(get_contributions(a)),args...)
end

function RBSteady.allocate_hypred_cache(a::NNContribution,args...)
  fecache = allocate_coefficient(a,args...)
  hypred = allocate_hyper_reduction(a,args...)
  return HRParamArray(fecache,fecache,hypred)
end

function FESpaces.interpolate!(
  hypred::AbstractArray,
  cache::AbstractArray,
  a::NNContribution,
  r::AbstractRealisation
  )

  fill!(hypred,zero(eltype(hypred)))
  for aval in get_contributions(a)
    interpolate!(hypred,cache,aval,r)
  end
  return hypred
end

# transient 

function RBSteady.HRProjection(
  red::TransientNNRegression,
  s::Snapshots,
  trian::Triangulation,
  test::RBSpace
  )

  r = get_realisation(s)
  b = GalerkinProjectable(s)
  y = galerkin_projection(test,b)
  ϕ = get_basis(y)
  model = train_neural_coefficient(get_strategy(red),r,ϕ)
  return NNRegressor(model,test)
end

function RBSteady.HRProjection(
  red::TransientNNRegression,
  s::Snapshots,
  trian::Triangulation,
  trial::RBSpace,
  test::RBSpace
  )

  r = get_realisation(s)
  A = GalerkinProjectable(s)
  y = galerkin_projection(test,A,trial,get_time_combination(red))
  ϕ = get_basis(y)
  model = train_neural_coefficient(get_strategy(red),r,ϕ)
  return NNRegressor(model,trial,test)
end

function RBSteady.HRProjection(
  red::TransientNNHyperReduction,
  s::Snapshots,
  trian::Triangulation,
  test::RBSpace
  )

  basis = projection(get_reduction(red),s)
  proj_basis = project(test,basis)
  interp = Interpolation(red,basis,s)
  return HRProjection(proj_basis,red,interp)
end

function RBSteady.HRProjection(
  red::TransientNNHyperReduction,
  s::Snapshots,
  trian::Triangulation,
  trial::RBSpace,
  test::RBSpace
  )

  basis = projection(get_reduction(red),s)
  proj_basis = project(test,basis,trial,get_time_combination(red))
  interp = Interpolation(red,basis,s)
  return HRProjection(proj_basis,red,interp)
end

const TransientNNRegressor{A<:Projection} = HRProjection{<:AbstractTransientNNHyperReduction,A}
const TransientNNContribution = AffineContribution{<:TransientNNRegressor}
const TransientNNContributionTuple = ContributionTuple{N,<:TransientNNContribution} where N

function FESpaces.interpolate!(
  b̂::AbstractArray,
  coeff::Tuple,
  a::TransientNNContributionTuple,
  r::AbstractRealisation
  )

  fill!(b̂,zero(eltype(b̂)))
  for (ai,ci) in zip(a,coeff)
    for aval in get_contributions(ai)
      interpolate!(b̂,ci,aval,r)
    end
  end
  return b̂
end

# NN 

function train_neural_coefficient(red::NeuralReduction,data...;normalise=true,kwargs...)
  (values,params),stats = get_inputs_and_stats(red,feop,data...;normalise)
  inputs = (values,nothing,params)
  train(red,inputs,stats;kwargs...)
end

# utils

_axpy!(α,a,b) = @abstractmethod

function _axpy!(α,a::AbstractMatrix,b::AbstractParamVector)
  axpy!(α,a,get_all_data(b))
end

function _axpy!(α,a::AbstractMatrix,b::AbstractParamMatrix)
  nrows,ncols = innersize(b)
  k = param_length(b)
  a′ = reshape(a,nrows,ncols,k)
  axpy!(α,a′,get_all_data(b))
end
