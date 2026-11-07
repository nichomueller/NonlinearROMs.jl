using Test
using LinearAlgebra
using Graphs
using Gridap
using Gridap.FESpaces
using GridapROMs
using NonlinearROMs

using NonlinearROMs: WeightedSimpleDiGraph,get_weight,out_weights,in_weights

@testset "WeightedSimpleDiGraph" begin
  g = WeightedSimpleDiGraph(4)

  @test nv(g) == 4
  @test ne(g) == 0
  @test is_directed(g)
  @test eltype(g) == Int
  @test collect(vertices(g)) == 1:4

  @test add_edge!(g,1,2,1.5)
  @test !add_edge!(g,1,2,1.5)     # duplicate edge: no-op, doesn't overwrite
  @test add_edge!(g,1,3,2.5,9.0)  # distinct forward/backward weights
  @test add_edge!(g,2,3,0.5)
  @test ne(g) == 3

  @test has_edge(g,1,2)
  @test has_edge(g,1,3)
  @test has_edge(g,2,3)
  @test !has_edge(g,2,1)          # directed: reverse edge was never added
  @test !has_edge(g,3,1)
  @test !has_edge(g,1,4)

  @test collect(outneighbors(g,1)) == [2,3]
  @test collect(inneighbors(g,3)) == [1,2]
  @test isempty(outneighbors(g,4))
  @test isempty(inneighbors(g,1))

  # forward weight is cached on the source, backward weight on the destination
  @test get_weight(g,1,2) == 1.5
  @test get_weight(g,1,3) == 2.5
  @test get_weight(g,2,3) == 0.5
  @test get_weight(g,3,1) == 0.0  # no such edge -> zero, not an error

  @test out_weights(g,1) == [1.5,2.5]
  @test in_weights(g,3) == [9.0,0.5]  # backward weight of (1, 3) was set to 9.0 explicitly
  @test in_weights(g,2) == [1.5]

  W = weights(g)
  @test W[1,2] == 1.5
  @test W[1,3] == 2.5
  @test W[2,3] == 0.5
  @test W[3,1] == 0.0

  @test rem_edge!(g,1,2)
  @test !has_edge(g,1,2)
  @test ne(g) == 2
  @test !rem_edge!(g,1,2)  # already removed

  @test add_vertex!(g)
  @test nv(g) == 5
  @test isempty(outneighbors(g,5))

  h = copy(g)
  @test h == g
  @test h !== g
  @test h.fadjlist !== g.fadjlist  # deep copy: mutating h doesn't affect g
  add_edge!(h,4,5,1.0)
  @test h != g

  # generic AbstractSimpleGraph fallback, exercised through add_edge!/rem_edge!
  nv_before = nv(g)
  @test rem_vertex!(g,1)
  @test nv(g) == nv_before-1
end

@testset "GraphsInterface & GraphInducedSpace" begin
  model = CartesianDiscreteModel((0,1,0,1),(6,6))
  reffe = ReferenceFE(lagrangian,Float64,1)
  V = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags="boundary")
  coords = get_coords(V)
  npts = length(coords)
  radius = 0.35

  @testset "build_graph & NystromSubgraph" begin
    g = build_graph(coords;radius=radius)
    @test nv(g) == npts
    @test ne(g) > 0

    for s in vertices(g)
      nbrs = outneighbors(g,s)
      ws = out_weights(g,s)
      @test length(nbrs) == length(ws)
      for (d,w) in zip(nbrs,ws)
        @test w <= radius + 1e-12
        @test w ≈ norm(coords[s] - coords[d])
      end
    end

    sub_n = min(10,npts)
    subg = NystromSubgraph(g;subgraph_size=sub_n)
    @test nv(subg.graph) == sub_n
    @test length(subg.vertex_map) == sub_n
  end

  @testset "GraphInducedSpace, gather_data, scatter_data & attention" begin
    gspace = GraphInducedSpace(V; radius=radius)
    g = get_graph(gspace)
  
    src_ids,dst_ids = NonlinearROMs.get_edge_ids(gspace)
    edge_weights = NonlinearROMs.get_edge_weights(gspace)
  
    # Edge indices and normalized weights
    @test length(src_ids) == ne(g)
    @test length(dst_ids) == ne(g)
    @test size(edge_weights) == (1,1,ne(g))
  
    @test all(i -> 1 <= i <= nv(g),src_ids)
    @test all(i -> 1 <= i <= nv(g),dst_ids)
  
    @test all(isfinite,edge_weights)
    @test minimum(edge_weights) >= 0.0f0
    @test maximum(edge_weights) ≈ 1.0f0 atol=1e-5
  
    # Every cached edge index must correspond to an actual graph edge
    for e_idx in eachindex(src_ids)
      @test has_edge(g,src_ids[e_idx],dst_ids[e_idx])
    end
  
    D,K = 4,3
    vals = rand(Float32,D,npts,K)
    gathered = gather_data(vals,gspace)
  
    @test size(gathered) == (2D + 1,ne(g),K)
  
    # Check source, destination and edge-weight features
    for e_idx in unique((1,max(1,ne(g) ÷ 2),ne(g)))
      s = src_ids[e_idx]
      d = dst_ids[e_idx]
      w = edge_weights[1,1,e_idx]
  
      @test gathered[1:D,e_idx,:] ≈ vals[:,s,:]
      @test gathered[(D+1):(2D),e_idx,:] ≈ vals[:,d,:]
      @test all(x -> isapprox(x,w; atol=1e-6),
                gathered[2D+1,e_idx,:])
    end
  
    edge_vals = ones(Float32,D,ne(g),K)
    scattered = scatter_data(edge_vals,gspace)
  
    @test size(scattered) == (D,npts,K)
    for v in vertices(g)
      in_deg = Float32(length(inneighbors(g,v)))
      @test all(x -> isapprox(x,in_deg; atol=1e-5),scattered[:,v,:])
    end
  
    scores = randn(Float32,ne(g))
    α = NonlinearROMs.attention(scores,gspace)
  
    @test length(α) == ne(g)
    @test all(isfinite,α)
    @test all(x -> x >= 0.0f0,α)
  
    # Incoming attention weights must sum to one for each non-isolated vertex
    for v in vertices(g)
      in_mask = findall(==(v),dst_ids)
      if !isempty(in_mask)
        @test sum(α[in_mask]) ≈ 1.0f0 atol=1e-5
      end
    end
  end
end