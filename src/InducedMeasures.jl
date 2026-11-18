# message passing

abstract type MPRule end

MPRule() = PlainMP()

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

function compute_weights(f::GraphInducedSpace)
  graph = get_graph(f)
  deg = zeros(Int,nv(graph))
  for d in get_dst_ids(f)
    deg[d] += 1
  end
  inv_deg = TData[d > 0 ? inv(TData(d)) : zero(TData) for d in deg]
  return reshape(inv_deg,1,nv(graph),1)
end

compute_weights(f::MultipleSpaces) = map(compute_weights,f.lspaces)

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

"""
    const MultipleMeasure = InducedMeasure{<:MultipleSpaces,<:Tuple}

Measure induced by a [`MultipleSpaces`](@ref) of [`FunctionSpace`](@ref)s or
[`GraphInducedSpace`](@ref)s, `weights` holding one entry per local space. Its edges
are those of the local measures (see [`get_local_measures`](@ref)), concatenated block
by block and relabelled with the global vertex ids, so `gather_data`/`kv` work on
the global, vertex-indexed data as for any other measure. Integration is carried out
block by block on the local measures, then
- summed over the blocks when the local spaces share their samples (multi-field);
- concatenated along the sample axis otherwise (one block of samples per local space,
  e.g. several PDEs on a moving geometry), so that every sample is integrated on its
  own local measure only. The samples are then tied to the blocks, and cannot be
  batched.
"""
const MultipleMeasure = InducedMeasure{<:MultipleSpaces,<:Tuple}

function get_local_measures(meas::MultipleMeasure)
  space = get_space(meas)
  rule = get_message_rule(meas)
  map((s,w) -> InducedMeasure(s,w,rule),space.lspaces,get_weights(meas))
end

num_vertices(meas::MultipleMeasure) = num_coords(get_space(meas))
num_edges(meas::MultipleMeasure) = sum(num_edges,get_local_measures(meas))

for F in (:get_src_ids,:get_dst_ids)
  @eval begin
    function $F(meas::MultipleMeasure)
      l_to_lcoord_to_coords = get_space(meas).l_to_lcoord_to_coords
      ids = map(enumerate(get_local_measures(meas))) do (l,m)
        l_to_lcoord_to_coords[l][$F(m)]
      end
      reduce(vcat,ids)
    end
  end
end

get_edge_weights(meas::MultipleMeasure) = reduce(vcat,map(get_edge_weights,get_local_measures(meas)))

function integration(a::AbstractInputData,meas::InducedMeasure)
  msg = "Integrand must be of size (latent_dim, num_edges, num_params), where num_edges == $(num_edges(meas)); got size $(size(a))"
  @check size(a,2) == num_edges(meas) msg
  _integration(a,meas)
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
    blockwise(f,meas::MultipleMeasure,edata::AbstractInputData...) -> AbstractInputData

Evaluates `f(ldata...,m)` on every local measure `m` of `meas`, where `ldata` are the
edge-indexed `edata` restricted to the edges (and, for sample-stacked spaces, to the
samples) of `m`, and `f` returns local vertex-indexed data. The local results are then
expanded onto the global vertices via `l_to_lcoord_to_coords` (a placement, these maps
being injective, built on `NNlib.scatter` on plain index arrays so that nothing is
unrolled per edge under a Reactant trace), and summed (shared samples) or concatenated
along the sample axis (one block of samples per local space).
"""
function blockwise(f,meas::MultipleMeasure,edata::AbstractInputData...)
  space = get_space(meas)
  l_to_sample_offset = space.l_to_sample_offset
  _check_samples(l_to_sample_offset,first(edata))
  lmeas = get_local_measures(meas)
  N = num_vertices(meas)
  eoff = 0
  outs = map(enumerate(lmeas)) do (l,m)
    edges = (eoff+1):(eoff+num_edges(m))
    eoff += num_edges(m)
    samples = _get_range(l_to_sample_offset,l)
    ldata = map(a -> a[:,edges,samples],edata)
    expand_vertex_data(f(ldata...,m),space.l_to_lcoord_to_coords[l],N)
  end
  _combine_blocks(l_to_sample_offset,outs)
end

function expand_vertex_data(ldata::AbstractInputData,lcoord_to_coords::AbstractVector,N::Int)
  d,_,k = size(ldata)
  l′ = permutedims(ldata,(1,3,2))
  out′ = NNlib.scatter(+,l′,collect(Int,lcoord_to_coords);dstsize=(d,k,N))
  permutedims(out′,(1,3,2))
end

_combine_blocks(::Nothing,outs) = reduce(+,outs)
_combine_blocks(::AbstractVector,outs) = cat(outs...;dims=3)

_check_samples(::Nothing,a) = nothing

function _check_samples(l_to_sample_offset::AbstractVector,a)
  msg = """The data has $(size(a,3)) samples, but the measure expects $(last(l_to_sample_offset)-1),
  one block per local space: the samples of a MultipleSpaces with sample offsets cannot be
  batched (use a full batch, `batch_size=0`)"""
  @check size(a,3) == last(l_to_sample_offset)-1 msg
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

_integration(a::AbstractInputData,meas::MultipleMeasure) = blockwise(_integration,meas,a)
  
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