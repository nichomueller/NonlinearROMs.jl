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
  @test in_weights(g,3) == [9.0,0.5]  # backward weight of (1,3) was set to 9.0 explicitly
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

@testset "GraphsInterface" begin
  model = CartesianDiscreteModel((0,1,0,1),(6,6))
  reffe = ReferenceFE(lagrangian,Float64,2)
  V = LexicographicFESpace(model,reffe;conformity=:H1,dirichlet_tags="boundary")
  coords = get_free_dof_coordinates(V)
  ndofs = length(coords)

  @testset "MeshGraph" begin
    g = build_graph(MeshGraph(),V)

    @test nv(g) == ndofs
    @test ne(g) > 0

    # every stored weight is the Euclidean distance between its two dofs
    for s in vertices(g)
      nbrs = outneighbors(g,s)
      ws = out_weights(g,s)
      @test length(nbrs) == length(ws)
      for (d,w) in zip(nbrs,ws)
        @test w ≈ norm(coords[s]-coords[d])
      end
    end

    # dof-sharing-a-cell adjacency is symmetric, even though each direction
    # is inserted independently while looping over cells
    for s in vertices(g), d in outneighbors(g,s)
      @test has_edge(g,d,s)
    end
  end

  @testset "DistanceGraph" begin
    radius = 0.3
    g = build_graph(DistanceGraph(radius),V)

    @test nv(g) == ndofs

    for s in vertices(g)
      nbrs = outneighbors(g,s)
      for d in nbrs
        w = get_weight(g,s,d)
        @test w <= radius
        @test w ≈ norm(coords[s]-coords[d])
      end
    end
  end
end