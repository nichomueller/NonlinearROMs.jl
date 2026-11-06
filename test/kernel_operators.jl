using Test
using LinearAlgebra
using Random
using Graphs
using Lux
using Gridap
using Gridap.FESpaces
using GridapROMs
using GridapROMs.ParamDataStructures
using GridapROMs.ParamFESpaces
using GridapROMs.ParamSteady
using GridapROMs.RBSteady
using GridapSolvers
using GridapSolvers.LinearSolvers
using NonlinearROMs

using NonlinearROMs: TData,num_coords,get_weights,_kv,build_model

@testset "Kernel Neural Operators" begin

  @testset "InducedMeasures & Integration" begin
    model = CartesianDiscreteModel((0,1,0,1),(6,6))
    reffe = ReferenceFE(lagrangian,Float64,1)
    V = TestFESpace(model,reffe; conformity=:H1,dirichlet_tags="boundary")

    # 1. GraphMeasure integration: integrating a constant message c across all incoming
    # edges of each node must yield exactly c at every node (degree-normalized average)
    gspace = GraphInducedSpace(V; radius=0.35)
    g = get_graph(gspace)
    gmeas = InducedMeasure(gspace)

    D,N,E,K = 4,num_coords(gspace),ne(g),3
    @test size(get_weights(gmeas)) == (1,N,1)

    c_val = 2.75f0
    edge_const = fill(c_val,D,E,K)
    node_int = integration(edge_const,gmeas)
    @test size(node_int) == (D,N,K)
    @test all(x -> isapprox(x,c_val; atol=1e-5),node_int)

    # 2. FESpaceMeasure integration & Sobolev Energy Losses
    fspace = FunctionSpace(V)
    fmeas = InducedMeasure(fspace)
    @test size(get_weights(fmeas)) == (N,N)

    u_rand = rand(TData,1,N,K)
    @test size(integration(u_rand,fmeas)) == (1,N,K)

    for loss_fn in (NonlinearROMs.L2Loss(fspace),NonlinearROMs.H1Loss(fspace),NonlinearROMs.H1L2Loss(fspace))
      @test loss_fn(u_rand,u_rand) ≈ 0.0f0 atol=1e-6
      @test loss_fn(u_rand,zeros(TData,1,N,K)) > 0.0f0
    end
  end

  @testset "Kernel Network Non-Linearity & _kv Modes" begin
    model = CartesianDiscreteModel((0,1,0,1),(5,5))
    reffe = ReferenceFE(lagrangian,Float64,1)
    V = TestFESpace(model,reffe; conformity=:H1,dirichlet_tags="boundary")
    gspace = GraphInducedSpace(V; radius=0.3)
    fspace = FunctionSpace(V)

    N = num_coords(gspace)
    in_dims,out_dims,K = 6,1,4
    dummy_in = rand(TData,in_dims,N,K)
    dummy_out = rand(TData,out_dims,N,K)

    # Test both channel-wise (diagonal) and full-matrix kernel products in _kv
    din,dout = 8,4
    v_test = rand(TData,din,20,3)
    k_diag = fill(2.0f0,din,20,3)
    @test _kv(k_diag,v_test) ≈ 2.0f0 .* v_test

    # Full-matrix kernel: each edge/point applies a matrix-vector product
    k_mat = rand(TData,dout * din,20,3)
    result = _kv(k_mat,v_test)
    @test size(result) == (dout,20,3)

    for e in 1:20,k in 1:3
      W = reshape(k_mat[:,e,k],dout,din)
      @test result[:,e,k] ≈ W * v_test[:,e,k]
    end

    # Verify model construction across both discrete spaces (Graph & FESpace)
    # and all kernel input formulations
    for space in (gspace,fspace)
      for fmt in (
        DataFormatter(pass_coords=Val(true),pass_features=Val(false),pass_new_data=Val(false)),
        DataFormatter(pass_coords=Val(true),pass_features=Val(true),pass_new_data=Val(false)),
        DataFormatter(pass_coords=Val(true),pass_features=Val(false),pass_new_data=Val(true)),
        DataFormatter(pass_coords=Val(true),pass_features=Val(true),pass_new_data=Val(true))
      )
        kno = KernelNeuralModel(
          space;
          hidden_layers=(8,8,8),
          activation=Lux.gelu,
          formatter=fmt
        )
        chain = build_model(kno,(dummy_in,dummy_out))

        # Lift + 2 KernelNeuralLayers + Project = 4 layers
        @test length(chain.layers) == 4
        klayer1 = chain.layers.layer_2

        # Both the integral kernel network κ_θ and the projection Q must be non-linear MLPs (Lux.Chain)
        @test klayer1.kernel.kernel isa Lux.Chain
        @test chain.layers.layer_4.layer isa Lux.Chain

        ps,st = Lux.setup(Random.default_rng(),chain)
        pred,_ = chain(dummy_in,ps,st)
        @test size(pred) == (out_dims,N,K)
        @test !any(isnan,pred)
      end
    end
  end

  @testset "End-to-End Training & Non-Mutating Inference" begin
    pdomain = (1.0,5.0,1.0,5.0)
    pspace = ParamSpace(pdomain)
    model = CartesianDiscreteModel((0,1,0,1),(6,6))
    Ω = Triangulation(model)
    dΩ = Measure(Ω,2)

    a(μ) = x -> μ[1]
    aμ(μ) = parameterise(a,μ)
    f(μ) = x -> μ[2] * sin(π * x[1]) * sin(π * x[2])
    fμ(μ) = parameterise(f,μ)
    g(μ) = x -> 0.0
    gμ(μ) = parameterise(g,μ)

    stiffness(μ,u,v) = ∫(aμ(μ) * ∇(v) ⋅ ∇(u))dΩ
    rhs(μ,v) = ∫(fμ(μ) * v)dΩ

    reffe = ReferenceFE(lagrangian,Float64,1)
    test_space = TestFESpace(Ω,reffe; conformity=:H1,dirichlet_tags="boundary")
    trial_space = ParamTrialFESpace(test_space,gμ)
    feop = LinearParamFEOperator(rhs,stiffness,pspace,trial_space,test_space)
    fesolver = LinearFESolver(LUSolver())

    μ_train = realisation(pspace; nparams=8,sampling=:halton)
    uh_train = solve(fesolver,feop,μ_train)

    x_coords = get_coords(test_space)
    in_train = InputSpace(x_coords,FeatureBundle((aμ(μ_train),fμ(μ_train))))
    out_train = OutputSpace(uh_train)

    coords_backup = copy(in_train.coords)
    features_backup = copy(in_train.features)
    values_backup = copy(out_train.values)

    gspace = GraphInducedSpace(test_space; radius=0.35)
    gno = KernelNeuralModel(gspace; hidden_layers=(8,8),activation=tanh)
    solver = NeuralSolver(gno; epochs=5,batch_size=4,verbose=false)

    op = neural_operator(solver,in_train,out_train)

    # Training must not mutate the user's InputSpace or OutputSpace tensors in place
    @test in_train.coords ≈ coords_backup
    @test in_train.features ≈ features_backup
    @test out_train.values ≈ values_backup

    # Repeated online solve calls on the same InputSpace must be idempotent
    μ_test = realisation(pspace; nparams=3,sampling=:halton,start=9)
    in_test = InputSpace(x_coords,FeatureBundle((aμ(μ_test),fμ(μ_test))))
    test_coords_backup = copy(in_test.coords)
    test_features_backup = copy(in_test.features)

    pred1 = solve(solver,op,in_test)
    pred2 = solve(solver,op,in_test)

    @test in_test.coords ≈ test_coords_backup
    @test in_test.features ≈ test_features_backup
    @test pred1 ≈ pred2
  end

end