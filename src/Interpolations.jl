"""
    struct NNInterpolation{A<:NeuralNetwork} <: Interpolation

An [`Interpolation`](@ref) backed by a neural network model. During the
online phase, `interpolate!` replaces the EIM linear solve with a NN forward
pass: `coeff[:,i] = model(μ_i)`.

Constructed automatically by `Interpolation(red::NNHyperReduction,basis,s)`.
"""
struct NNInterpolation{A<:NeuralNetwork} <: Interpolation
  interpolation::A
end

function RBSteady.Interpolation(
  red::NNHyperReduction,
  a::Projection,
  s::Snapshots
  )

  inds,interp = DEIM(a)
  factor = lu(interp)
  r = get_realisation(s)
  red_data = RBSteady.get_at_domain(s,inds)
  coeff = parameterise(allocate_in_domain(a),r)
  ldiv!(coeff,factor,red_data)
  model = train_neural_coefficient(get_strategy(red),r,coeff)
  NNInterpolation(model)
end

# transient 

function RBSteady.Interpolation(
  red::TransientNNHyperReduction,
  a::KroneckerProjection,
  s::TransientSnapshots
  )

  inds,interp = DEIM(a)
  factor = lu(interp)
  r = get_params(get_realisation(s))
  red_data = RBTransient.get_at_kron_domain(s,inds...)
  coeff = parameterise(allocate_in_domain(a),r)
  ldiv!(coeff,factor,red_data)
  model = train_neural_coefficient(get_strategy(red),r,coeff)
  NNInterpolation(model)
end

function RBSteady.Interpolation(
  red::TransientNNHyperReduction,
  a::SequentialProjection,
  s::TransientSnapshots
  )

  inds,interp = DEIM(a)
  factor = lu(interp)
  r = get_params(get_realisation(s))
  red_data = RBTransient.get_at_seq_domain(s,inds...)
  coeff = parameterise(allocate_in_domain(a),r)
  ldiv!(coeff,factor,red_data)
  model = train_neural_coefficient(get_strategy(red),r,coeff)
  NNInterpolation(model)
end