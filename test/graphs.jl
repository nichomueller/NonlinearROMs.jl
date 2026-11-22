using Test
using LinearAlgebra
using Graphs
using Gridap
using Gridap.FESpaces
using GridapROMs
using NonlinearROMs

using NonlinearROMs: WeightedSimpleDiGraph,get_weight,out_weights,in_weights,TData,num_coords

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
    
    # Exact distance radius must give uniform interior in-degree without Float64 edge drops
    model_lat = CartesianDiscreteModel((0,1,0,1),(10,10))
    V_lat = TestFESpace(model_lat,reffe;conformity=:H1)
    coords_lat = get_coords(V_lat)
    g_lat = build_graph(coords_lat;radius=2 / 10)
    int_ids = findall(p -> 0.2 <= p[1] <= 0.8 && 0.2 <= p[2] <= 0.8,coords_lat)
    @test all(v -> length(inneighbors(g_lat,v)) == 13,int_ids)
  end

  @testset "GraphInducedSpace, gather_data, scatter_data & attention" begin
    gspace = GraphInducedSpace(V; radius=radius)
    g = get_graph(gspace)

    # gather_data/scatter_data/attention run on an InducedMeasure, not a bare space --
    # PlainMP is requested explicitly here (vcat(src,dst,weight)) since the default
    # message_rule is AnchoredMP, which this test's shape/content assertions don't match.
    gmeas = InducedMeasure(gspace; message_rule=PlainMP())

    src_ids = NonlinearROMs.get_src_ids(gmeas)
    dst_ids = NonlinearROMs.get_dst_ids(gmeas)
    edge_weights = NonlinearROMs.get_edge_weights(gmeas)

    # Edge indices and normalized weights
    @test length(src_ids) == ne(g)
    @test length(dst_ids) == ne(g)
    @test length(edge_weights) == ne(g)

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
    gathered = gather_data(vals,gmeas)

    @test size(gathered) == (2D + 1,ne(g),K)

    # Check source, destination and edge-weight features
    for e_idx in unique((1,max(1,ne(g) ÷ 2),ne(g)))
      s = src_ids[e_idx]
      d = dst_ids[e_idx]
      w = edge_weights[e_idx]

      @test gathered[1:D,e_idx,:] ≈ vals[:,s,:]
      @test gathered[(D+1):(2D),e_idx,:] ≈ vals[:,d,:]
      @test all(x -> isapprox(x,w; atol=1e-6),
                gathered[2D+1,e_idx,:])
    end

    edge_vals = ones(Float32,D,ne(g),K)
    scattered = scatter_data(edge_vals,gmeas)

    @test size(scattered) == (D,npts,K)
    for v in vertices(g)
      in_deg = Float32(length(inneighbors(g,v)))
      @test all(x -> isapprox(x,in_deg; atol=1e-5),scattered[:,v,:])
    end

    scores = randn(Float32,ne(g))
    α = NonlinearROMs.attention(scores,gmeas)

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
  
  @testset "MPRules (DifferenceMP, CompositeMP, PlainMP, AnchoredMP)" begin
    model = CartesianDiscreteModel((0,1,0,1),(5,5))
    reffe = ReferenceFE(lagrangian,Float64,1)
    V = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags="boundary")
    gspace = GraphInducedSpace(V;radius=0.35)
    g = get_graph(gspace)
    src_ids = NonlinearROMs.get_src_ids(gspace)
    dst_ids = NonlinearROMs.get_dst_ids(gspace)
    w = NonlinearROMs.get_edge_weights(gspace)
  
    D,N,K = 3,num_coords(gspace),2
    vals = rand(TData,D,N,K)
  
    # DifferenceMP: src - dst (D channels)
    g_diff = gather_data(vals,InducedMeasure(gspace;message_rule=DifferenceMP()))
    @test size(g_diff) == (D,ne(g),K)
    @test g_diff[:,1,:] ≈ vals[:,src_ids[1],:] .- vals[:,dst_ids[1],:]
  
    # CompositeMP: [src - dst; dist] (D + 1 channels)
    g_comp = gather_data(vals,InducedMeasure(gspace;message_rule=CompositeMP()))
    @test size(g_comp) == (D + 1,ne(g),K)
    @test g_comp[1:D,1,:] ≈ vals[:,src_ids[1],:] .- vals[:,dst_ids[1],:]
    @test all(x -> isapprox(x,w[1];atol=1e-6),g_comp[D+1,1,:])
  
    # PlainMP: [src; dst; dist] (2D + 1 channels)
    g_plain = gather_data(vals,InducedMeasure(gspace;message_rule=PlainMP()))
    @test size(g_plain) == (2D + 1,ne(g),K)
    
    # AnchoredMP: [dst; src - dst; dist] (2D + 1 channels)
    g_anch = gather_data(vals,InducedMeasure(gspace;message_rule=AnchoredMP()))
    @test size(g_anch) == (2D + 1,ne(g),K)
    @test g_anch[1:D,1,:] ≈ vals[:,dst_ids[1],:]
    @test g_anch[(D+1):(2D),1,:] ≈ vals[:,src_ids[1],:] .- vals[:,dst_ids[1],:]
    @test all(x -> isapprox(x,w[1];atol=1e-6),g_anch[2D+1,1,:])
  end
end