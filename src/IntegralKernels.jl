abstract type KernelLayer <: Lux.AbstractLuxLayer end

function Lux.initialstates(rng::Random.AbstractRNG,l::KernelLayer)
  return (
    kernel = NamedTuple(),
  )
end

struct NeuralKernel{A<:NeuralModel} <: KernelLayer
  model::A
end

function Lux.initialparameters(rng::Random.AbstractRNG,l::NeuralKernel)
  return Lux.initialparameters(rng,l.model)
end

function (l::NeuralKernel)(x,ps,st)
  l.model(x,ps,st)
end

abstract type AbstractIntegralKernel <: Lux.AbstractLuxLayer end

# we potentially want to replace the space with a new one during the online phase
# thus we make this struct mutable
mutable struct IntegralKernel{A<:KernelLayer,B<:InducedMeasure} <: AbstractIntegralKernel
  const kernel::A
  measure::B
end

function Lux.initialparameters(rng::Random.AbstractRNG,l::IntegralKernel)
  Lux.initialparameters(rng,l.kernel)
end

function Lux.initialstates(rng::Random.AbstractRNG,l::IntegralKernel)
  Lux.initialstates(rng,l.kernel)
end

function (l::IntegralKernel)(x,ps,st)
  # vals is Nh x dout, where Nh is to be integrated over (next line)
  vals,st = l.kernel(x,ps,st)
  y = integrate_over(vals,l.measure)
  return y,st
end

struct FourierKernel <: AbstractIntegralKernel
  in_channels::Int
  out_channels::Int
  truncation::Int
end

function Lux.initialparameters(rng::Random.AbstractRNG,l::FourierKernel)
  return (
    kernel = rand(rng,Uniform(-1,1),l.truncation,l.out_channels,l.in_channels),
  )
end

function (l::FourierKernel)(x,ps,st)
  x̂ = rfft(x)
  ŷ = fapply(ps.kernel,x̂)
  y = irfft(ŷ)
  return y,st
end

# utils 

function fapply(W::AbstractArray{T,3},x::AbstractMatrix) where T
  k,din,dout = size(W)
  A = reshape(permutedims(W,(2,1,3)),k*dout,din)
  b = vec(x)
  reshape(A*b,k,dout)
end