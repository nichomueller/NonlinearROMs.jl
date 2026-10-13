# integrate over an array

function integrate_over(a::AbstractArray{T,N},space) where {T,N}
  msg = "The array's final dimension must match the number of coordinates in the function space"
  @check size(a,N) == dimension(space) msg
  _integrate_over(a,space)
end

function _integrate_over(a::AbstractArray{T,N},space) where {T,N}
  mat = reshape(a,:,size(a,N))
  slices = eachslice(mat,dims=2)
  s,state = iterate(slices)
  int = _integrate_over(s,space)
  @inbounds for s in Iterators.rest(slices,state)
    int .+= _integrate_over(s,space)
  end
  return int
end

function _integrate_over(a::AbstractVector,space)
  @abstractmethod
end

struct InducedMeasure{A} 
  space::A
end

function integrate_over(a::AbstractArray,meas::InducedMeasure)
  integrate_over(a,meas.space)
end

# function spaces

dimension(f::FunctionSpace) = num_coords(f)

# on a simple space, we can dispatch to Gridap's quadrature rules
function _integrate_over(a::AbstractVector,f::SimpleFESpace)
  integrate(FEFunction(f.space,a),f)
end

function _integrate_over(a::AbstractArray{T,N},f::MultiSpace) where {T,N}
  ccache = array_cache(f.lcoord_to_coords)
  lcache = array_cache(f.coord_to_lcoords)
  int = zeros(eltype(a),dimension(f))
  @inbounds for i in eachindex(f.spaces)
    ccoords = getindex!(ccache,f.lcoord_to_coords,i)
    lcoords = getindex!(lcache,f.coord_to_lcoords,i)
    weighted = a[lcoords] ./ f.multiplicity[lcoords]
    int += _integrate_over(weighted,f.spaces[i])
  end
  return int
end

# graphs

# graphs are evaluated per-edge, then aggregated per-vertex
dimension(g::AbstractGraph) = ne(g)
dimension(g::NystromSubgraphIterator) = dimension(g.graph)

"""
    _integrate_over(a,g::AbstractGraph) -> per-vertex array

Mean-aggregates `a` (one column per edge) over each vertex's incoming edges:
`out[...,i] = mean_{e: dst(e)=i} a[...,e]`, i.e. the Monte-Carlo/graph-kernel
discretization of `(K v)(x_i) ≈ (1/|N(i)|) Σ_{j∈N(i)} κ(x_i,x_j,...) v(x_j)`. Returns
an array with last-axis size `nv(g)` -- one value per vertex, not one pooled scalar
over the whole graph.
"""
function _integrate_over(a::AbstractVector,g::AbstractGraph)
  out = scatter(a,g)
  cnt = zeros(Int,nv(g))
  for e in edges(g)
    cnt[dst(e)] += 1
  end
  for i in eachindex(cnt)
    cnt[i] > 0 && (out[i] /= cnt[i])
  end
  return out
end

function _integrate_over(a::AbstractMatrix,g::AbstractGraph)
  out = scatter(a,g)
  cnt = zeros(Int,nv(g))
  for e in edges(g)
    cnt[dst(e)] += 1
  end
  for i in axes(out,2)
    cnt[i] > 0 && (@views out[:,i] ./= cnt[i])
  end
  return out
end

function _integrate_over(a,g::NystromSubgraphIterator)
  @notimplemented """
  Nystrom-sampled graph integration needs the kernel re-evaluated on a freshly
  resampled edge subset each round -- that's the whole point of Nystrom sampling,
  avoiding the cost of evaluating the kernel on every edge -- not a different
  reduction applied to one precomputed `vals` tensor. That means this has to be
  driven from `IntegralKernel`'s forward pass (resample -> build the subgraph's
  edge input -> evaluate the kernel on just that -> repeat), not from
  `_integrate_over` alone, which only ever sees one already-computed `vals` array.
  Not implemented yet. (For the record, the previous version here was also wrong on
  its own terms: `NystromSubgraphIterator` iterates `(subgraph,vmap)` tuples, not
  bare graphs -- see `sample_subgraph`/`GraphLayer`'s `NystromGraphLayer` in
  NeuralLayers.jl, which destructures it correctly -- and summing per-round
  aggregates without tracking how many rounds actually touched each vertex would
  double/under-count regardless.)
  """
end

"""
    gather_edge_features(v,g::AbstractGraph;extra=nothing) -> (...,ne(g)) array

Assembles the per-edge kernel input `[v(sender);v(receiver)]` (optionally with a
static per-vertex feature array `extra`, e.g. a PDE coefficient `a`, concatenated
in as `[v(sender);v(receiver);extra(sender);extra(receiver)]`), in `edges(g)` order
-- i.e. it builds the `x` that makes `κ(x,y,v(x),v(y))` (or, with `extra`,
`κ(x,y,a(x),a(y),v(x),v(y))`) a genuine per-edge, both-endpoints-dependent
evaluation, which `scatter`/`_integrate_over` then aggregate back down to
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
prepare_kernel_input(v,meas::InducedMeasure) = prepare_kernel_input(v,meas.space)

# utils 

Base.@propagate_inbounds function _get_channel(a::AbstractArray{T,N},c::Int) where {T,N}
  @boundscheck 1 <= c <= size(a,N)
  @views vec(a[Base.Cartesian.@nexprs N-1 _ -> :,c])
end

Base.@propagate_inbounds function _set_channel!(a::AbstractArray{T,N},v,c::Int) where {T,N}
  @boundscheck 1 <= c <= size(a,N)
  @views a[Base.Cartesian.@nexprs N-1 _ -> :,c] .= v
end