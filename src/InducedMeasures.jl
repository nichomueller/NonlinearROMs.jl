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
  out = fill!(similar(a),zero(eltype(a)))
  for (i,m) in enumerate(meas)
    ar = restrict_edge_data(space,a,i)
    outr = integration(ar,m)
    out += expand_vertex_data(space,outr,i)
  end
  return out
end

"""
    restrict_edge_data(space::MultiGraphInducedSpace,gdata::AbstractInputData,l::Int=1)

Gathers the global, vertex-indexed `gdata` (one column per point in
`space`'s shared union of coordinates) onto block `l`'s own local graph
edges (one column per edge, each holding its source vertex's global data,
through `space.l_to_lcoord_to_coords[l]`'s local-to-global index map) --
local counterpart of [`gather_data`](@ref), and built the same way, via
`NNlib.gather` on plain (non-traced) integer index arrays computed from the
graph structure alone, rather than a per-edge `setindex!` loop: at a few
thousand edges per block (routine for a radius graph), such a loop gets
unrolled into one MLIR op per edge, which is exactly what `gather_data`'s
own docstring warns can exhaust Reactant's tracing budget.
"""
function restrict_edge_data(
  space::MultiGraphInducedSpace,
  gdata::AbstractInputData,
  l::Int=1
  )

  lgraph = get_graph(space[l])
  lcoord_to_coords = space.l_to_lcoord_to_coords[l]

  ids = lcoord_to_coords[src.(edges(lgraph))]
  g′ = permutedims(gdata,(1,3,2))
  ldata′ = NNlib.gather(g′,ids)
  permutedims(ldata′,(1,3,2))
end

"""
    expand_vertex_data(space::MultiGraphInducedSpace,ldata::AbstractInputData,l::Int=1)

Scatters block `l`'s local, vertex-indexed `ldata` (the result of integrating
over block `l`'s own local graph) back onto `space`'s shared global vertex
indexing, through `space.l_to_lcoord_to_coords[l]`'s local-to-global index
map (injective, so this amounts to a placement rather than an accumulation)
-- local counterpart of [`scatter_data`](@ref), and built the same way, via
`NNlib.scatter` on a plain (non-traced) index array, for the same
Reactant-tracing-budget reason as [`restrict_edge_data`](@ref).
"""
function expand_vertex_data(
  space::MultiGraphInducedSpace,
  ldata::AbstractInputData,
  l::Int=1
  )

  ggraph = get_graph(space)
  lcoord_to_coords = space.l_to_lcoord_to_coords[l]

  d,_,k = size(ldata)
  l′ = permutedims(ldata,(1,3,2))
  out′ = NNlib.scatter(+,l′,lcoord_to_coords;dstsize=(d,k,nv(ggraph)))
  permutedims(out′,(1,3,2))
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