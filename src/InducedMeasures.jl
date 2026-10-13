dimension(f::FunctionSpace) = num_coords(f)
dimension(g::AbstractGraph) = ne(g) # graphs are evaluated per-edge

abstract type InducedMeasure end

CellData.integrate(a,meas::InducedMeasure) = integration(a,meas)

integration(a,meas::InducedMeasure) = @abstractmethod

get_space(meas::InducedMeasure) = @abstractmethod

function integration(a::AbstractArray{T,N},meas::InducedMeasure) where {T,N}
  latent...,dim = size(a)
  integration(reshape(a,prod(latent),dim),meas)
end

function integration(a::AbstractVecOrMat,meas::InducedMeasure)
  msg = "The array's final dimension must match the number of coordinates in the function space"
  dim = size(a,ndims(a))
  @check dim == dimension(get_space(meas)) msg
  _integration(a,meas)
end

function _integration(a::AbstractVecOrMat,meas::InducedMeasure)
  @abstractmethod
end

struct FSpaceMeasure{A<:FunctionSpace} <: InducedMeasure
  space::A
end

InducedMeasure(space::FunctionSpace) = FSpaceMeasure(space)
InducedMeasure(space::MultiSpace{CommonDomain}) = FSpaceMeasure(build_graph(space))

get_space(meas::FSpaceMeasure) = meas.space

function _integration(a::AbstractVector,meas::FSpaceMeasure{<:SimpleFESpace})
  space = get_space(meas)
  ah = FEFunction(space,a)
  integrate(ah,space)
end

function _integration(a::AbstractMatrix,meas::FSpaceMeasure{<:SimpleFESpace})
  int = zeros(eltype(a),size(a,1))
  @inbounds for col in eachcol(a)
    int .+= _integration(col,meas)
  end
  return int
end

function _integration(a::AbstractVector,meas::FSpaceMeasure{<:MultiSpace})
  space = get_space(meas)
  lcache = array_cache(space.coord_to_lcoords)
  int = zeros(eltype(a))
  @inbounds @views for i in eachindex(space.spaces)
    lcoords = getindex!(lcache,space.coord_to_lcoords,i)
    weighted = a[lcoords] ./ space.multiplicity[lcoords]
    int += _integration(weighted,space.spaces[i])
  end
  return int
end

function _integration(a::AbstractMatrix,meas::FSpaceMeasure{<:MultiSpace})
  space = get_space(meas)
  ccache = array_cache(space.lcoord_to_coords)
  lcache = array_cache(space.coord_to_lcoords)
  int = zeros(eltype(a),size(a,1))
  @inbounds @views for i in eachindex(space.spaces)
    ccoords = getindex!(ccache,space.lcoord_to_coords,i)
    lcoords = getindex!(lcache,space.coord_to_lcoords,i)
    weighted = a[:,lcoords] ./ space.multiplicity[lcoords]
    int[ccoords] += _integration(weighted,space.spaces[i])
  end
  return int
end

struct GraphMeasure{A<:AbstractGraph,B<:AbstractVector} <: InducedMeasure
  graph::A
  weights::B
end

function InducedMeasure(graph::AbstractGraph)
  w = zeros(Int,nv(graph))
  for e in edges(graph)
    w[dst(e)] += 1
  end
  return GraphMeasure(graph,w)
end

get_space(meas::GraphMeasure) = meas.graph

function _integration(a::AbstractArray,meas::GraphMeasure)
  graph = get_space(meas)
  out = scatter(a,graph)
  _normalise!(out,meas.weights)
  return out
end

"""
    gather_edge_features(v,g::AbstractGraph;extra=nothing) -> (...,ne(g)) array

Assembles the per-edge kernel input `[v(sender);v(receiver)]` (optionally with a
static per-vertex feature array `extra`, e.g. a PDE coefficient `a`, concatenated
in as `[v(sender);v(receiver);extra(sender);extra(receiver)]`), in `edges(g)` order
-- i.e. it builds the `x` that makes `κ(x,y,v(x),v(y))` (or, with `extra`,
`κ(x,y,a(x),a(y),v(x),v(y))`) a genuine per-edge, both-endpoints-dependent
evaluation, which `scatter`/`_integration` then aggregate back down to
one value per vertex. `v`/`extra` are `(channels,n_vertices)`, matching the
Lux/`NeuralModel` `(channels,batch)` convention.
"""
function gather_edge_features(v::AbstractMatrix,g::AbstractGraph;extra=nothing)
  feats = isnothing(extra) ? v : vcat(v,extra)
  din = size(feats,1)
  out = zeros(eltype(feats),2*din,ne(g))
  for (k,e) in enumerate(edges(g))
    @views out[1:din,k] .= feats[:,src(e)]
    @views out[din+1:end,k] .= feats[:,dst(e)]
  end
  return out
end

"""
    prepare_kernel_input(v,space) -> whatever the kernel network expects

Converts the raw hidden state `v` (`(channels,n_points)`, values at *every* point
of the function/graph `v` lives on) into whatever `num_coords(space)`-indexed input
a `KernelLayer` needs to be evaluated on, dispatching on `space`:
- FE-quadrature (`FunctionSpace`): identity -- `v` already is the nodal values at
  that space's own coordinates, so no assembly is needed.
- Graph: `gather_edge_features(v,space)` -- `κ(x,y,...)` depending on both
  endpoints needs the per-edge `[v(sender);v(receiver)]` pair, not `v` itself.

This is what lets `KernelNeuralLayer` call every `AbstractIntegralKernel` uniformly
with the same raw `v`, regardless of which non-local discretization it uses.
"""
prepare_kernel_input(v,f::FunctionSpace) = v
prepare_kernel_input(v,g::AbstractGraph) = gather_edge_features(v,g)
prepare_kernel_input(v,meas::InducedMeasure) = prepare_kernel_input(v,get_space(meas))

# utils 

function _normalise!(a::AbstractVector,weights::AbstractVector)
  for i in eachindex(a)
    weights[i] > 0 && (a[i] /= weights[i])
  end
  a
end

function _normalise!(a::AbstractMatrix,weights::AbstractVector)
  @views for i in eachindex(a)
    weights[i] > 0 && (a[:,i] ./= weights[i])
  end
  a
end