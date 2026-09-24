const NNHRProjection{A<:AbstractNNHyperReduction,B<:Projection} = HRProjection{A,B}

function FESpaces.interpolate!(
  b̂::AbstractArray,
  cache,
  a::NNHRProjection{<:NNHyperReduction,<:Projection},
  r::AbstractRealisation
  )

  o = one(eltype2(b̂))
  x = matrix_of_params(r)
  i = get_interpolation(a)
  coeff = ConsecutiveParamArray(evaluate!(cache,i.interpolation,x))
  mul!(b̂,a,coeff,o,o)
  return b̂
end

struct NNOperator{A,B} <: NNHRProjection{NNOperatorReduction,B}
  model::A
  bias::B
end

function NNOperator(model::NeuralModel,test::RBSpace)
  T = get_dof_value_type(test)
  nrows = num_reduced_dofs(test)
  basis = ReducedProjection(zeros(T,nrows,1))
  NNOperator(model,basis)
end

function NNOperator(model::NeuralModel,trial::RBSpace,test::RBSpace)
  T = get_dof_value_type(trial)
  nrows = num_reduced_dofs(test)
  ncols = num_reduced_dofs(trial)
  basis = ReducedProjection(zeros(T,nrows,ncols,1))
  NNOperator(model,basis)
end

RBSteady.get_basis(a::NNOperator) = a.bias
RBSteady.get_style(a::NNOperator) = NNOperatorReduction()
RBSteady.get_interpolation(a::NNOperator) = EmptyInterpolation()
RBSteady.projection_eltype(a::NNOperator) = RBSteady.projection_eltype(a.bias)

function FESpaces.interpolate!(
  b̂::AbstractArray,
  cache,
  a::NNOperator,
  r::AbstractRealisation
  )

  b̂r = evaluate!(cache,a.model,matrix_of_params(r))
  o = one(eltype2(b̂))
  _axpy!(o,b̂r,b̂)
  return b̂
end

function RBSteady.HRProjection(
  red::NNOperatorReduction,
  s::Snapshots,
  trian::Triangulation,
  test::RBSpace
  )

  r = get_realisation(s)
  b = GalerkinProjectable(s)
  y = galerkin_projection(test,b)
  ϕ = get_basis(y)
  model = train_neural_coefficient(get_strategy(red),r,ϕ)
  return NNOperator(model,test)
end

function RBSteady.HRProjection(
  red::NNOperatorReduction,
  s::Snapshots,
  trian::Triangulation,
  trial::RBSpace,
  test::RBSpace
  )

  r = get_realisation(s)
  A = GalerkinProjectable(s)
  y = galerkin_projection(test,A,trial)
  ϕ = get_basis(y)
  model = train_neural_coefficient(get_strategy(red),r,ϕ)
  return NNOperator(model,trial,test)
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

function RBSteady.allocate_coefficient(a::NNOperator,r::AbstractRealisation)
  x = matrix_of_params(r)
  return_cache(a.model,x)
end

function RBSteady.allocate_coefficient(a::NNHRProjection{<:NNHyperReduction,<:Projection},r::AbstractRealisation)
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
  coeffs = fecache
  hypred = allocate_hyper_reduction(a,args...)
  return HRParamArray(fecache,coeffs,hypred)
end

function FESpaces.interpolate!(
  hypred::AbstractArray,
  cache,
  a::NNContribution,
  r::AbstractRealisation
  )

  fill!(hypred,zero(eltype(hypred)))
  for aval in get_contributions(a)
    interpolate!(hypred,cache,aval,r)
  end
  return hypred
end

function RBSteady.allocate_coefficient(
  a::BlockHRProjection{<:AbstractNNHyperReduction,<:Any,<:Any,N},
  r::AbstractRealisation
  ) where N

  i0 = findfirst(a.touched)
  A = typeof(allocate_coefficient(a.array[i0],r))
  block_cache = Array{A,N}(undef,size(a))
  for i in eachindex(a)
    if a.touched[i]
      block_cache[i] = allocate_coefficient(a.array[i],r)
    end
  end
  return ArrayBlock(block_cache,a.touched)
end

# transient 

function RBSteady.HRProjection(
  red::TransientNNOperatorReduction,
  s::Snapshots,
  trian::Triangulation,
  test::RBSpace
  )

  r = get_realisation(s)
  b = GalerkinProjectable(s)
  y = galerkin_projection(test,b)
  ϕ = get_basis(y)
  model = train_neural_coefficient(get_strategy(red),r,ϕ)
  return NNOperator(model,test)
end

function RBSteady.HRProjection(
  red::TransientNNOperatorReduction,
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
  return NNOperator(model,trial,test)
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

const TransientNNProjection{A<:Projection} = HRProjection{<:AbstractTransientNNHyperReduction,A}
const TransientNNContribution = AffineContribution{<:TransientNNProjection}
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
