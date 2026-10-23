dimension(f::FunctionSpace) = num_coords(f)
dimension(g::AbstractGraph) = ne(g) # graphs are evaluated per-edge
dimension(space::AbstractGraphInducedSpace) = dimension(get_graph(space))

function compute_weights(f::DiscreteSpace)
  @abstractmethod
end

function compute_weights(f::FunctionSpace{<:FESpace})
  dΩ = get_measure(f)
  M = assemble_matrix((u,v) -> ∫(u⋅v)dΩ,f.space,f.space)
  return TData.(M)
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
  graph = get_graph(f)
  w = zeros(Int,nv(graph))
  cache = array_cache(f.node_to_lnode)
  for e in edges(graph)
    v = dst(e)
    lnodes = getindex!(cache,f.node_to_lnode,v)
    w[v] += length(lnodes)
  end
  return w
end

struct InducedMeasure{A<:DiscreteSpace,B<:AbstractArray}
  space::A
  weights::B
end

function InducedMeasure(space::DiscreteSpace)
  weights = compute_weights(space)
  InducedMeasure(space,weights)
end

const FESpaceMeasure = InducedMeasure{<:FunctionSpace{<:FESpace},<:AbstractMatrix}
const GraphMeasure = InducedMeasure{<:GraphInducedSpace,<:AbstractVector}

integration(a,meas::InducedMeasure) = @abstractmethod

get_space(meas::InducedMeasure) = meas.space
get_weights(meas::InducedMeasure) = meas.weights

function integration(a::InputData,meas::InducedMeasure)
  msg = "Integrand must be of size (latent_dim, num_coords, num_params), where num_coords == $(dimension(get_space(meas))); got size $(size(a))"
  @check size(a,2) == dimension(get_space(meas)) msg
  _integration(a,meas)
end

function _integration(a::AbstractArray,meas::InducedMeasure)
  @notimplemented
end

function _integration(a::InputData,meas::FESpaceMeasure)
  M = get_weights(meas)
  D,N,K = size(a)
  a2 = reshape(permutedims(a,(2,1,3)),N,D*K)
  b2 = M*a2
  permutedims(reshape(b2,N,D,K),(2,1,3))
end

get_graph(meas::GraphMeasure) = get_graph(get_space(meas))

function _integration(a::InputData,meas::GraphMeasure)
  space = get_space(meas)
  out = scatter_data(a,space)
  _normalise!(out,meas.weights)
  return out
end

# utils 

function _normalise!(a::InputData,weights::AbstractVector)
  @views for i in axes(a,2)
    !iszero(weights[i]) && (a[:,i,:] ./= weights[i])
  end
  a
end