"""
    abstract type NeuralNetwork <: Map end

Abstract supertype for neural network models. Any concrete subtype must
implement `(a::T)(x::AbstractMatrix) -> AbstractMatrix` where `x` is a
`(param_dim × batch_size)` matrix of parameters and the output is a
`(output_dim × batch_size)` matrix of predictions.

Wrap external Flux/Lux models with [`GenericNeuralNetwork`](@ref):

    model = GenericNeuralNetwork(flux_chain)     # Flux
    model = GenericNeuralNetwork(p -> lux_apply(chain,p,ps,st))  # Lux closure
"""
abstract type NeuralNetwork <: Map end


"""
    abstract type AbstractFiniteDimensionalNetwork <: NeuralNetwork end

Abstract supertype for standard neural networks mapping between finite-dimensional Euclidean spaces (Vector -> Vector).
"""
abstract type AbstractFiniteDimensionalNetwork <: NeuralNetwork end

"""
    abstract type AbstractNeuralOperator <: NeuralNetwork end

Abstract supertype for neural operators, which learn mappings between infinite-dimensional function spaces (Function -> Function).
"""
abstract type AbstractNeuralOperator <: NeuralNetwork end

"""
    abstract type AbstractCoordinateBasedOperator <: AbstractNeuralOperator end

Abstract supertype for neural operators that evaluate the solution point-by-point using continuous physical coordinates.
"""
abstract type AbstractCoordinateBasedOperator <: AbstractNeuralOperator end

"""
    abstract type AbstractKernelNeuralOperator <: AbstractNeuralOperator end

Abstract supertype for neural operators based on iterative kernel integration.
These models process the entire spatial field or graph simultaneously rather than point-by-point.
"""
abstract type AbstractKernelNeuralOperator <: AbstractNeuralOperator end

"""
    struct GenericNeuralNetwork{A} <: NeuralNetwork
      model::A
    end

Wraps any callable `A` (Flux chain, Lux closure, plain Julia function) as
an [`NeuralNetwork`](@ref). The wrapped callable must accept a
`(d × k)` parameter matrix and return an `(n × k)` prediction matrix.
"""
struct GenericNeuralNetwork{A} <: NeuralNetwork
  model::A
end

function Arrays.return_cache(a::GenericNeuralNetwork,x::AbstractMatrix)
  return_cache(a.model,x)
end

function Arrays.evaluate!(cache,a::GenericNeuralNetwork,x::AbstractMatrix)
  evaluate!(cache,a.model,x)
end

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

_get_data(a) = get_all_data(a)
_get_data(a::AbstractParamMatrix) = reshape(get_all_data(a),innerlength(a),:)
_get_data(a::AbstractMatrix) = a
_get_data(a::AbstractArray{T,3}) where T = reshape(a,:,size(a,3))