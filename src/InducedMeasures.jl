dimension(f::FunctionSpace) = num_coords(f)
dimension(g::AbstractGraph) = ne(g) # graphs are evaluated per-edge
dimension(space::AbstractGraphInducedSpace) = dimension(get_graph(space))

function compute_weights(f::DiscreteSpace)
  @abstractmethod
end

function compute_weights(f::FunctionSpace{<:FESpace})
  dΩ = get_measure(f)
  M = assemble_matrix((u,v) -> ∫(u⋅v)dΩ,f.space,f.space)
  m = get_node_masks(f)
  return TData.(M[m,m])
end

function compute_weights(f::GraphInducedSpace)
  graph = get_graph(f)
  w = zeros(Int,nv(graph))
  for e in edges(graph)
    w[dst(e)] += 1
  end
  return w
end

function compute_weights(f::MultiGraphInducedSpace)
  w = ()
  for fi in f
    w = (w...,compute_weights(fi))
  end
  return w
end

struct InducedMeasure{A<:DiscreteSpace,B}
  space::A
  weights::B
end

function InducedMeasure(space::DiscreteSpace)
  weights = compute_weights(space)
  InducedMeasure(space,weights)
end

const FESpaceMeasure = InducedMeasure{<:FunctionSpace{<:FESpace},<:AbstractMatrix}
const GraphMeasure = InducedMeasure{<:GraphInducedSpace,<:AbstractVector}
const MultiGraphMeasure = InducedMeasure{<:MultiGraphInducedSpace,<:Tuple}

integration(a,meas::InducedMeasure) = @abstractmethod
get_space(meas::InducedMeasure) = meas.space
get_weights(meas::InducedMeasure) = meas.weights

get_graph(meas::GraphMeasure) = get_graph(get_space(meas))

function integration(a::AbstractInputData,meas::InducedMeasure)
  msg = "Integrand must be of size (latent_dim, num_coords, num_params), where num_coords == $(dimension(get_space(meas))); got size $(size(a))"
  @check size(a,2) == dimension(get_space(meas)) msg
  _integration(a,meas)
end

function Base.iterate(meas::MultiGraphMeasure,i::Int=1)
  space = get_space(meas)
  i > length(space) && return nothing
  (InducedMeasure(space[i],get_weights(meas)[i]),i+1)
end

function integration(a::AbstractInputData,meas::MultiGraphMeasure)
  space = get_space(meas)
  D,_,K = size(a)
  N = num_coords(space)
  out = zeros(eltype(a),D,N,K)
  for (i,m) in enumerate(meas)
    ar = restrict_data(space,a,i)
    outr = integration(ar,m)
    out += expand_data(space,outr,i)
  end
  return out
end

# utils

function _integration(a::AbstractArray,meas::InducedMeasure)
  @notimplemented
end

function _integration(a::AbstractInputData,meas::FESpaceMeasure)
  M = get_weights(meas)
  _mat_mul_data(M,a)
end

function _integration(a::AbstractInputData,meas::GraphMeasure)
  w = get_weights(meas)
  space = get_space(meas)
  out = scatter_data(a,space)
  @views for i in axes(out,2)
    !iszero(w[i]) && (out[:,i,:] ./= w[i])
  end
  return out
end

function _mat_mul_data(A::AbstractMatrix,x::AbstractInputData)
  D,N,K = size(x)
  x2 = reshape(permutedims(x,(2,1,3)),N,D*K)
  b2 = A*x2
  permutedims(reshape(b2,N,D,K),(2,1,3))
end