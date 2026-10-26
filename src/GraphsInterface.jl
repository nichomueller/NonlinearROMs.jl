"""
    WeightedSimpleDiGraph{Ti<:Integer,Tv<:Real} <: Graphs.AbstractSimpleGraph{Ti}

A directed graph with the same internal layout as `Graphs.SimpleDiGraph{Ti}`
(a pair of sorted forward/backward adjacency lists), extended with a scalar
weight attached to every directed edge from both endpoints' perspectives:

- `fweights[s][k]` is the weight of the edge `s -> fadjlist[s][k]`, stored
  alongside `fadjlist` for O(1) access when iterating `outneighbors(g,s)`.
- `bweights[d][k]` is the weight of the edge `badjlist[d][k] -> d`, stored
  alongside `badjlist` for O(1) access when iterating `inneighbors(g,d)`.

By default `add_edge!` stores the same value on both sides of an edge; pass
distinct forward/backward weights explicitly for asymmetric couplings (e.g.
upwinded or non-symmetric operators on a mesh).
"""
mutable struct WeightedSimpleDiGraph{Ti<:Integer,Tv<:Real} <: Graphs.AbstractSimpleGraph{Ti}
  ne::Int
  fadjlist::Vector{Vector{Ti}}
  badjlist::Vector{Vector{Ti}}
  fweights::Vector{Vector{Tv}}
  bweights::Vector{Vector{Tv}}

  function WeightedSimpleDiGraph{Ti,Tv}(
    ne::Int,
    fadjlist::Vector{Vector{Ti}},
    badjlist::Vector{Vector{Ti}},
    fweights::Vector{Vector{Tv}},
    bweights::Vector{Vector{Tv}}
    ) where {Ti<:Integer,Tv<:Real}

    Graphs.SimpleGraphs.throw_if_invalid_eltype(Ti)
    return new{Ti,Tv}(ne,fadjlist,badjlist,fweights,bweights)
  end
end

function WeightedSimpleDiGraph(
  ne::Int,
  fadjlist::Vector{Vector{Ti}},
  badjlist::Vector{Vector{Ti}},
  fweights::Vector{Vector{Tv}},
  bweights::Vector{Vector{Tv}}
  ) where {Ti<:Integer,Tv<:Real}

  return WeightedSimpleDiGraph{Ti,Tv}(ne,fadjlist,badjlist,fweights,bweights)
end

function WeightedSimpleDiGraph{Ti,Tv}(n::Integer=0) where {Ti<:Integer,Tv<:Real}
  fadjlist = [Vector{Ti}() for _ in 1:n]
  badjlist = [Vector{Ti}() for _ in 1:n]
  fweights = [Vector{Tv}() for _ in 1:n]
  bweights = [Vector{Tv}() for _ in 1:n]
  return WeightedSimpleDiGraph{Ti,Tv}(0,fadjlist,badjlist,fweights,bweights)
end

WeightedSimpleDiGraph(n::Ti) where {Ti<:Integer} = WeightedSimpleDiGraph{Ti,Float64}(n)
WeightedSimpleDiGraph() = WeightedSimpleDiGraph{Int,Float64}()

function WeightedSimpleDiGraph(g::SimpleDiGraph{Ti},fw::Tv=1.0,bw::Tv=fw) where {Ti<:Integer,Tv<:Real}
  mg = WeightedSimpleDiGraph{Ti,Tv}(nv(g))
  for e in edges(g)
    add_edge!(mg,src(e),dst(e),fw,bw)
  end
  return mg
end

# core AbstractSimpleGraph interface

Graphs.edgetype(::WeightedSimpleDiGraph{Ti}) where {Ti} = Edge{Ti}
Graphs.is_directed(::Type{<:WeightedSimpleDiGraph}) = true

Graphs.SimpleGraphs.badj(g::WeightedSimpleDiGraph) = g.badjlist
Graphs.SimpleGraphs.badj(g::WeightedSimpleDiGraph,v::Integer) = g.badjlist[v]

function Base.copy(g::WeightedSimpleDiGraph{Ti,Tv}) where {Ti,Tv}
  WeightedSimpleDiGraph{Ti,Tv}(
    g.ne,
    Graphs.deepcopy_adjlist(g.fadjlist),
    Graphs.deepcopy_adjlist(g.badjlist),
    Graphs.deepcopy_adjlist(g.fweights),
    Graphs.deepcopy_adjlist(g.bweights)
  )
end

function Base.:(==)(g::WeightedSimpleDiGraph,h::WeightedSimpleDiGraph)
  (
    vertices(g) == vertices(h) &&
    ne(g) == ne(h) &&
    g.fadjlist == h.fadjlist &&
    g.badjlist == h.badjlist &&
    g.fweights == h.fweights &&
    g.bweights == h.bweights
  )
end

function Graphs.has_edge(g::WeightedSimpleDiGraph{Ti},e::Edge{Ti}) where {Ti}
  s,d = Ti.(Tuple(e))
  verts = vertices(g)
  (s in verts && d in verts) || return false
  @inbounds list = g.fadjlist[s]
  @inbounds list_backedge = g.badjlist[d]
  if length(list) > length(list_backedge)
    d = s
    list = list_backedge
  end
  return Graphs.insorted(d,list)
end

# weighted edge insertion/removal

function Graphs.add_edge!(g::WeightedSimpleDiGraph{Ti,Tv},e::Edge{Ti},fw::Tv,bw::Tv=fw) where {Ti,Tv}
  s,d = Ti.(Tuple(e))
  verts = vertices(g)
  (s in verts && d in verts) || return false

  @inbounds flist = g.fadjlist[s]
  findex = searchsortedfirst(flist,d)
  @inbounds (findex <= length(flist) && flist[findex] == d) && return false
  insert!(flist,findex,d)
  insert!(g.fweights[s],findex,fw)

  g.ne += 1

  @inbounds blist = g.badjlist[d]
  bindex = searchsortedfirst(blist,s)
  insert!(blist,bindex,s)
  insert!(g.bweights[d],bindex,bw)

  return true
end

function Graphs.add_edge!(g::WeightedSimpleDiGraph{Ti,Tv},e::Edge{Ti}) where {Ti,Tv}
  return add_edge!(g,e,one(Tv))
end

function Graphs.add_edge!(g::WeightedSimpleDiGraph{Ti,Tv},s::Integer,d::Integer,fw::Tv,bw::Tv=fw) where {Ti,Tv}
  return add_edge!(g,Edge{Ti}(Ti(s),Ti(d)),fw,bw)
end

function Graphs.rem_edge!(g::WeightedSimpleDiGraph{Ti},e::Edge{Ti}) where {Ti}
  s,d = Ti.(Tuple(e))
  verts = vertices(g)
  (s in verts && d in verts) || return false

  @inbounds flist = g.fadjlist[s]
  findex = searchsortedfirst(flist,d)
  @inbounds (findex <= length(flist) && flist[findex] == d) || return false
  deleteat!(flist,findex)
  deleteat!(g.fweights[s],findex)

  g.ne -= 1

  @inbounds blist = g.badjlist[d]
  bindex = searchsortedfirst(blist,s)
  deleteat!(blist,bindex)
  deleteat!(g.bweights[d],bindex)

  return true
end

function Graphs.add_vertex!(g::WeightedSimpleDiGraph{Ti,Tv}) where {Ti,Tv}
  (nv(g) + one(Ti) <= nv(g)) && return false
  push!(g.fadjlist,Vector{Ti}())
  push!(g.badjlist,Vector{Ti}())
  push!(g.fweights,Vector{Tv}())
  push!(g.bweights,Vector{Tv}())
  return true
end

# weight accessors

"""
    get_weight(g::WeightedSimpleDiGraph,s,d) -> Real

Forward weight of the edge `s -> d` (i.e. the value stored in
`fweights[s]`), or `zero(Tv)` if the edge does not exist.
"""
function get_weight(g::WeightedSimpleDiGraph{Ti,Tv},s::Integer,d::Integer) where {Ti,Tv}
  verts = vertices(g)
  (s in verts && d in verts) || return zero(Tv)
  @inbounds list = g.fadjlist[s]
  index = searchsortedfirst(list,d)
  @inbounds (index <= length(list) && list[index] == d) || return zero(Tv)
  return @inbounds g.fweights[s][index]
end

get_weight(g::WeightedSimpleDiGraph,e::Edge) = get_weight(g,src(e),dst(e))

"""
    out_weights(g::WeightedSimpleDiGraph,v) -> Vector

Forward weights of the outgoing edges of `v`, in the same order as
`outneighbors(g,v)`.
"""
out_weights(g::WeightedSimpleDiGraph,v::Integer) = g.fweights[v]

"""
    in_weights(g::WeightedSimpleDiGraph,v) -> Vector

Backward weights of the incoming edges of `v`, in the same order as
`inneighbors(g,v)`.
"""
in_weights(g::WeightedSimpleDiGraph,v::Integer) = g.bweights[v]

function Graphs.weights(g::WeightedSimpleDiGraph{Ti,Tv}) where {Ti,Tv}
  n = nv(g)
  I = Ti[]
  J = Ti[]
  V = Tv[]
  for s in vertices(g)
    for (k,d) in enumerate(g.fadjlist[s])
      push!(I,s)
      push!(J,d)
      push!(V,g.fweights[s][k])
    end
  end
  return sparse(I,J,V,n,n)
end

struct NystromSubgraph{Ti,A<:AbstractGraph{Ti},B<:AbstractVector{<:Integer}} <: Graphs.AbstractSimpleGraph{Ti}
  graph::A
  vertex_map::B
end

function NystromSubgraph(rng::AbstractRNG,g::AbstractGraph;subgraph_size=100)
  i = randperm(rng,nv(g))
  indices = view(i,1:subgraph_size)
  g′,vmap = induced_subgraph(g,indices)
  NystromSubgraph(g′,vmap)
end

function NystromSubgraph(g::AbstractGraph;kwargs...)
  NystromSubgraph(Random.default_rng(),g;kwargs...)
end

Graphs.edgetype(::NystromSubgraph{Ti}) where {Ti} = Edge{Ti}
Graphs.is_directed(g::NystromSubgraph) = is_directed(g.graph)

Graphs.SimpleGraphs.badj(g::NystromSubgraph) = SimpleGraphs.badj(g.graph)
Graphs.SimpleGraphs.badj(g::NystromSubgraph,v::Integer) = SimpleGraphs.badj(g.graph,v)

function Base.copy(g::NystromSubgraph)
  NystromSubgraph(copy(g.graph),copy(g.vertex_map))
end

function Base.:(==)(g::NystromSubgraph,h::NystromSubgraph)
  g.graph == h.graph && g.vertex_map == h.vertex_map
end

function Graphs.has_edge(g::NystromSubgraph{Ti},e::Edge{Ti}) where {Ti}
  has_edge(g.graph,e)
end

function Graphs.add_edge!(g::NystromSubgraph{Ti},e::Edge{Ti},fw::Tv,bw::Tv=fw) where {Ti,Tv}
  add_edge!(g.graph,e,fw,bw)
end

function Graphs.add_edge!(g::NystromSubgraph{Ti},e::Edge{Ti}) where {Ti}
  add_edge!(g.graph,e)
end

function Graphs.add_edge!(g::NystromSubgraph{Ti},s::Integer,d::Integer,fw::Tv,bw::Tv=fw) where {Ti,Tv}
  add_edge!(g.graph,s,d,fw,bw)
end

function Graphs.rem_edge!(g::NystromSubgraph{Ti},e::Edge{Ti}) where {Ti}
  rem_edge!(g.graph,e)
end

function Graphs.add_vertex!(g::NystromSubgraph)
  add_vertex!(g.graph)
end

function Graphs.weights(g::NystromSubgraph)
  weights(g.graph)
end

# utils

function build_graph(coords::AbstractVector;kwargs...)
  data = map(x -> SVector(Tuple(x)),coords)
  tree = BallTree(data)
  g = WeightedSimpleDiGraph(length(coords))
  build_graph!(g,tree,coords;kwargs...)
  return GraphInducedSpace(g,coords)
end

function build_graph!(g::AbstractGraph,tree::NNTree,coords::AbstractVector;radius=1.0)
  for (i,coord) in enumerate(coords)
    q = SVector(Tuple(coord))
    neighbors = inrange(tree,q,radius)
    for neighbor in neighbors
      w = norm(coord - coords[neighbor])
      add_edge!(g,i,neighbor,w)
    end
  end
  g
end

# message-passing on graph and graph-induced spaces

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
"""
function gather_data(vals::InputData,g::AbstractGraph)
  @check size(vals,2) == nv(g) "The final dimension must match the number of graph vertices"
  idx_src = src.(edges(g))
  idx_dst = dst.(edges(g))
  vcat(vals[:,idx_src,:],vals[:,idx_dst,:])
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
destination vertex), so it's expressed as a dense `(nv,ne)` incidence-matrix matmul
(`_scatter_apply`) rather than a per-edge loop: a loop here gets unrolled into one set
of MLIR ops per edge -- fine at boh.jl's ~200 edges, but a multi-field graph (e.g.
Stokes' combined velocity+pressure point cloud) can have tens of thousands, enough to
exhaust Reactant's internal MLIR symbol-naming budget.

This standalone method builds `S` (`_scatter_matrix`) fresh every call, which is only
safe called *outside* an active Reactant/Enzyme trace: `S`'s own construction is a
`setindex!`-based loop, and once nested inside an already-tracing call stack (as it
would be if called from a model's forward pass during training), Reactant's overlay
intercepts even allocations/writes that don't depend on any traced value, turning `S`
into a traced array and hitting "scalar indexing disallowed". `GraphInducedSpace`
instead precomputes `S` once at construction (plain, untraced Julia, see
`DiscreteSpaces.jl`) and overrides `scatter_data` to reuse it -- that cached-matrix
path is the one actually exercised during training.
"""
function scatter_data(vals::InputData,g::AbstractGraph)
  @check size(vals,2) == ne(g) "The final dimension must match the number of graph edges"
  S = _scatter_matrix(eltype(vals),g)
  _scatter_apply(vals,S)
end

_scatter_matrix(::Type{T},g::AbstractGraph) where T = _scatter_matrix!(zeros(T,nv(g),ne(g)),g)

function _scatter_matrix!(S::AbstractMatrix,g::AbstractGraph)
  @inbounds for (i,e) in enumerate(edges(g))
    S[dst(e),i] = one(eltype(S))
  end
  S
end

function _scatter_apply(vals::InputData,S::AbstractMatrix)
  d,ne_,k = size(vals)
  n = size(S,1)
  v2 = reshape(permutedims(vals,(2,1,3)),ne_,d*k)
  out2 = S*v2
  permutedims(reshape(out2,n,d,k),(2,1,3))
end

"""
    attention(scores::AbstractVector,g::AbstractGraph) -> per-edge weights

Normalizes `scores` (one per edge) into attention weights
"""
function attention(scores::AbstractVector{T},g::AbstractGraph) where T
  @check length(scores) == ne(g) "The number of scores must match the number of graph edges"
  dsts = zeros(Int,length(edges(g)))
  for (k,e) in enumerate(edges(g))
    dsts[k] = dst(e)
  end

  maxs = fill(T(-Inf),nv(g))
  for (k,d) in enumerate(dsts)
    maxs[d] = max(maxs[d],scores[k])
  end

  α = similar(scores)
  sums = zeros(T,nv(g))
  for (k,d) in enumerate(dsts)
    α[k] = exp(scores[k]-maxs[d])
    sums[d] += α[k]
  end
  for (k,d) in enumerate(dsts)
    α[k] /= sums[d]
  end

  return α
end