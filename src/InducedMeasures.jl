# message passing

abstract type MPRule end

MPRule() = DifferenceMP()

combine_message(::MPRule,src,dst,args...) = @abstractmethod

function combine_message(r::MPRule,src,dst;edge_weights=nothing)
  isnothing(edge_weights) && return combine_message(r,src,dst)
  w = repeat(reshape(edge_weights,1,1,:);inner=(1,size(src,2),1))
  combine_message(r,src,dst,w)
end

struct PlainMP <: MPRule end

combine_message(::PlainMP,src,dst,args...) = vcat(src,dst,args...)

struct DifferenceMP <: MPRule end

combine_message(::DifferenceMP,src,dst,args...) = src .- dst

struct CompositeMP <: MPRule end

function combine_message(::CompositeMP,src,dst,args...)
  combine_message(DifferenceMP(),src,dst)
end

function combine_message(::CompositeMP,src,dst,w)
  combine_message(PlainMP(),combine_message(DifferenceMP(),src,dst),w)
end

# measures

function compute_weights(f::DiscreteSpace)
  @abstractmethod
end

function compute_weights(f::FunctionSpace{<:FESpace})
  assemble_norm(L2(),f)
end

function compute_weights(f::FunctionSpace{<:MultiFieldFESpace})
  nfields = num_fields(f.space)
  op = BlockOperator(ntuple(_ -> L2(),nfields))
  assemble_norm(op,f)
end

function compute_weights(f::GraphInducedSpace)
  graph = get_graph(f)
  deg = zeros(Int,nv(graph))
  for d in get_dst_ids(f)
    deg[d] += 1
  end
  inv_deg = TData[d > 0 ? inv(TData(d)) : zero(TData) for d in deg]
  return reshape(inv_deg,1,nv(graph),1)
end

function compute_weights(f::MultiGraphInducedSpace)
  w = ()
  for fi in f
    w = (w...,compute_weights(fi))
  end
  return w
end

struct InducedMeasure{A<:DiscreteSpace,B,C<:MPRule}
  space::A
  weights::B
  message_rule::C
end

function InducedMeasure(space::DiscreteSpace;message_rule::MPRule=MPRule())
  weights = compute_weights(space)
  InducedMeasure(space,weights,message_rule)
end

function InducedMeasure(space::DiscreteSpace,weights;message_rule::MPRule=MPRule())
  InducedMeasure(space,weights,message_rule)
end

integration(a,meas::InducedMeasure) = @abstractmethod
get_space(meas::InducedMeasure) = meas.space
get_weights(meas::InducedMeasure) = meas.weights
get_message_rule(meas::InducedMeasure) = meas.message_rule

get_graph(meas::InducedMeasure) = get_graph(get_space(meas))
num_vertices(meas::InducedMeasure) = nv(get_graph(meas))
num_edges(meas::InducedMeasure) = ne(get_graph(meas))

const FESpaceMeasure = InducedMeasure{<:FunctionSpace{<:FESpace},<:AbstractSparseMatrix}

num_vertices(meas::FESpaceMeasure) = size(get_weights(meas),1)
num_edges(meas::FESpaceMeasure) = nnz(get_weights(meas))
get_src_ids(meas::FESpaceMeasure) = findnz(get_weights(meas))[2]
get_dst_ids(meas::FESpaceMeasure) = findnz(get_weights(meas))[1]
get_edge_weights(meas::FESpaceMeasure) = nonzeros(get_weights(meas))

const GraphMeasure = InducedMeasure{<:GraphInducedSpace,<:AbstractArray}

get_src_ids(meas::GraphMeasure) = get_src_ids(get_space(meas))
get_dst_ids(meas::GraphMeasure) = get_dst_ids(get_space(meas))
get_edge_weights(meas::GraphMeasure) = get_edge_weights(get_space(meas))

const MultiGraphMeasure = InducedMeasure{<:MultiGraphInducedSpace,<:Tuple}

function integration(a::AbstractInputData,meas::InducedMeasure)
  msg = "Integrand must be of size (latent_dim, num_edges, num_params), where num_edges == $(num_edges(meas)); got size $(size(a))"
  @check size(a,2) == num_edges(meas) msg
  _integration(a,meas)
end

function Base.iterate(meas::MultiGraphMeasure,i::Int=1)
  space = get_space(meas)
  i > length(space) && return nothing
  m = InducedMeasure(space[i],get_weights(meas)[i];message_rule=get_message_rule(meas))
  (m,i+1)
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
    gather_data(vals,g::AbstractGraph) -> per-edge array

Pure indexing (`src.(edges(g))`/`dst.(edges(g))` are computed from `g` alone, never
from `vals`, so they're ordinary concrete arrays even when `vals` is a Reactant-traced
tensor): no edge ever needs to accumulate more than one value, unlike `scatter_data`,
so a plain vectorised gather suffices and no per-edge Julia loop is traced. A loop
here would otherwise get unrolled into one set of MLIR ops per edge -- fine at
boh.jl's ~200 edges, but a multi-field graph (e.g. Stokes' combined velocity+pressure
point cloud) can have tens of thousands, enough to exhaust Reactant's internal
MLIR symbol-naming budget.

Each edge's output column is the source node's features stacked on top of the
destination node's (channel axis doubles, `2*size(vals,1)`), not a separate output
column per endpoint (`NNlib.gather`, unlike `scatter`, never accumulates, so there's
no `op`/`dstsize` to combine src/dst into one pass -- two separate gathers are needed,
one per endpoint, then `vcat`'d along the channel axis).
"""
function gather_data(vals::AbstractInputData,meas::InducedMeasure)
  @check size(vals,2) == num_vertices(meas) "The final dimension must match the number of graph vertices"

  ids_src = get_src_ids(meas)
  ids_dst = get_dst_ids(meas)
  edge_weights = get_edge_weights(meas)
  rule = get_message_rule(meas)

  v′ = permutedims(vals,(1,3,2))
  src′ = NNlib.gather(v′,ids_src)
  dst′ = NNlib.gather(v′,ids_dst)
  out′ = combine_message(rule,src′,dst′;edge_weights)
  permutedims(out′,(1,3,2))
end

"""
    scatter_data(vals,g::AbstractGraph) -> per-vertex array

Sums `vals` (one column per edge, in `edges(g)` order; leading axes = channels) into
its destination vertex: `out[...,dst(e_k)] += vals[...,k]`. The shared building block
behind both the uniform-mean aggregation used by `_integration(a,::AbstractGraph)`
(plain GNO) and the softmax-weighted aggregation in `attention`/`AttentionKernel`
(HAMLET): the only difference between the two is what's summed -- `vals` itself for a
fixed `1/degree` weight, or `α .* vals` for a learned, data-dependent one.

Unlike `gather_data`, this genuinely needs to accumulate (several edges can share a
destination vertex), so a plain vectorised `getindex`-style gather doesn't suffice --
`NNlib.scatter(+,...)` is used instead (the same primitive `GNNGraphs.jl` builds its
own message passing on), rather than either a per-edge loop (gets unrolled into one
set of MLIR ops per edge under a Reactant/Enzyme trace -- fine at boh.jl's ~200 edges,
but a multi-field graph, e.g. Stokes' combined velocity+pressure point cloud, can have
tens of thousands, enough to exhaust Reactant's internal MLIR symbol-naming budget) or
the previous dense `(nv,ne)` incidence-matrix matmul (whose own construction was a
`setindex!`-based loop with the same problem, needing a separate precomputed-and-cached
path to stay off the traced call stack entirely). `NNlib.scatter`'s own edge axis is
the array's *last* dimension, not `vals`'s second (see `AbstractInputData`'s
`(D,N,K)` layout), hence the `permutedims` wrapping either side of the call.
"""
function scatter_data(vals::AbstractInputData,meas::InducedMeasure)
  @check size(vals,2) == num_edges(meas) "The number of columns in `vals` must match the number of graph edges"
  ids_dst = get_dst_ids(meas)
  d,_,k = size(vals)
  v′ = permutedims(vals,(1,3,2))
  out′ = NNlib.scatter(+,v′,ids_dst;dstsize=(d,k,num_vertices(meas)))
  permutedims(out′,(1,3,2))
end

"""
    attention(scores::AbstractVector,g::AbstractGraph) -> per-edge weights

Normalizes `scores` (one per edge) into attention weights: a numerically-stable
softmax over the scores of every edge sharing the same destination vertex (a
per-vertex group, not a single global softmax over all edges).

Same `NNlib.scatter` approach as `scatter_data`, applied to the two group
reductions a segmented softmax needs (max-per-group, then sum-per-group of the
shifted exponentials) instead of three sequential per-edge scalar loops, for the
same Reactant/MLIR-scaling reason. `scores`/`dsts` are already 1-D (one entry per
edge, no channel/sample axes), so no `permutedims` wrapping is needed here, unlike
`gather_data`/`scatter_data`.
"""
function attention(scores::AbstractVector{T},meas::InducedMeasure) where T
  @check length(scores) == num_edges(meas) "The number of scores must match the number of graph edges"
  ids_dst = get_dst_ids(meas)
  maxs = NNlib.scatter(max,scores,ids_dst;dstsize=(num_vertices(meas),),init=T(-Inf))
  α = exp.(scores .- maxs[ids_dst])
  sums = NNlib.scatter(+,α,ids_dst;dstsize=(num_vertices(meas),))
  α ./= sums[ids_dst]
  return α
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
  w = get_edge_weights(meas)
  scatter_data(a.*reshape(w,1,:,1),meas)
end

function _integration(a::AbstractInputData,meas::GraphMeasure)
  w = get_weights(meas)
  scatter_data(a,meas) .* w
end
  
function _mat_mul_data(A::AbstractMatrix,x::AbstractInputData)
  D,N,K = size(x)
  x2 = reshape(permutedims(x,(2,1,3)),N,D*K)
  b2 = A*x2
  permutedims(reshape(b2,N,D,K),(2,1,3))
end

# Specialized for sparse matrices to avoid dense matrix conversion during Reactant/XLA compilation
function _mat_mul_data(A::AbstractSparseMatrix,x::AbstractInputData)
  D,N,K = size(x)
  rows,cols,vals = findnz(A)
  w = reshape(vals,1,1,length(vals))
  x_perm = permutedims(x,(1,3,2))
  gathered = NNlib.gather(x_perm,cols)
  weighted = gathered .* w
  out_perm = NNlib.scatter(+,weighted,rows;dstsize=(D,K,size(A,1)))
  return permutedims(out_perm,(1,3,2))
end