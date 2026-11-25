using Test
using LinearAlgebra
using Random
using SparseArrays
using Statistics
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

using NonlinearROMs: TData,NormStyle,MultipleMeasure,num_coords,num_edges,get_weights,_kv,build_model,assemble_norm,get_formatter

@testset "Kernel Neural Operators" begin

  @testset "InducedMeasures & Integration" begin
    model = CartesianDiscreteModel((0,1,0,1),(6,6))
    reffe = ReferenceFE(lagrangian,Float64,1)
    V = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags="boundary")

    # 1. GraphMeasure integration: integrating a constant message c across all incoming
    # edges of each node must yield exactly c at every node (degree-normalized average)
    gspace = GraphInducedSpace(V;radius=0.35)
    g = get_graph(gspace)
    gmeas = InducedMeasure(gspace)

    D,N,E,K = 4,num_coords(gspace),ne(g),3
    @test size(get_weights(gmeas)) == (1,N,1)

    c_val = 2.75f0
    edge_const = fill(c_val,D,E,K)
    node_int = integration(edge_const,gmeas)
    @test size(node_int) == (D,N,K)
    @test all(x -> isapprox(x,c_val;atol=1e-5),node_int)

    # 2. FESpaceMeasure integration & Sobolev Energy Losses
    fspace = FunctionSpace(V)
    fmeas = InducedMeasure(fspace)
    @test size(get_weights(fmeas)) == (N,N)

    u_rand = rand(TData,1,N,K)
    M = get_weights(fmeas)
    _,cols,_ = findnz(M)
    edge_rand = u_rand[:,cols,:]
    fe_int = integration(edge_rand,fmeas)
    @test size(fe_int) == (1,N,K)
    Mu = stack(k -> M * u_rand[1,:,k],1:K;dims=2)
    @test fe_int ≈ reshape(Mu,1,N,K)

    for loss_fn in (EnergyLoss(ℓ2(),fspace),EnergyLoss(L2(),fspace),EnergyLoss(H1(),fspace))
      @test loss_fn(u_rand,u_rand) ≈ 0.0f0 atol=1e-6
      @test loss_fn(u_rand,zeros(TData,1,N,K)) > 0.0f0
    end

    # EnergyLoss must match the sample-averaged squared Gram norm (1/K) Σ_k e_k' X e_k
    u_zero = zeros(TData,1,N,K)
    X_l2 = assemble_norm(L2(),fspace)
    X_h1 = assemble_norm(H1(),fspace)
    expected_l2 = mean(dot(u_rand[1,:,k],X_l2 * u_rand[1,:,k]) for k in 1:K)
    expected_h1 = mean(dot(u_rand[1,:,k],X_h1 * u_rand[1,:,k]) for k in 1:K)
    @test isapprox(EnergyLoss(L2(),fspace)(u_rand,u_zero),expected_l2;rtol=1e-4)
    @test isapprox(EnergyLoss(H1(),fspace)(u_rand,u_zero),expected_h1;rtol=1e-4)

    # EnergyLoss converts Float64 sparse Gram matrices to TData without type promotion
    loss_f64 = EnergyLoss(Float64.(X_h1))
    val_f64 = loss_f64(u_rand,u_zero)
    @test val_f64 isa TData
    @test isapprox(val_f64,expected_h1;rtol=1e-4)

    # EnergyLoss accepts dense matrices by converting them to sparse format
    loss_dense = EnergyLoss(Matrix(X_h1))
    val_dense = loss_dense(u_rand,u_zero)
    @test val_dense isa TData
    @test isapprox(val_dense,expected_h1;rtol=1e-4)
  end

  @testset "Kernel Network Non-Linearity & _kv Modes" begin
    model = CartesianDiscreteModel((0,1,0,1),(5,5))
    reffe = ReferenceFE(lagrangian,Float64,1)
    V = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags="boundary")
    gspace = GraphInducedSpace(V;radius=0.3)
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

    # Verify model construction across both discrete spaces and all kernel input formulations
    for space in (gspace,fspace)
      for fmt in (SpatialKernelFormatter(),FeatureKernelFormatter(),StateKernelFormatter(),FullKernelFormatter())
        for cw in (true,false)
          kno = KernelNeuralModel(
            space;
            hidden_layers=(8,8,8),
            kernel_layers=(12,),
            channel_wise=cw,
            activation=Lux.gelu,
            formatter=fmt
          )
          chain = build_model(kno,(dummy_in,dummy_out))

          # Lift + 2 KernelNeuralLayers + Project = 4 layers
          @test length(chain.layers) == 4
          klayer1 = chain.layers.layer_2

          # Both the integral kernel network κ_θ and the projection Q must be non-linear MLPs
          @test klayer1.kernel.kernel isa Lux.Chain
          @test chain.layers.layer_4.layer isa Lux.Chain

          ps,st = Lux.setup(Random.default_rng(),chain)
          pred,_ = chain(dummy_in,ps,st)
          @test size(pred) == (out_dims,N,K)
          @test !any(isnan,pred)
        end
      end
    end

    # NeuralSolver must propagate norm_style and sampler from KernelNeuralModel's formatter
    ns_custom = NormStyle(;output_style=MScoreStyle())
    gno_ns = KernelNeuralModel(gspace;formatter=FeatureKernelFormatter(;norm_style=ns_custom))
    slv_ns = NeuralSolver(gno_ns;epochs=1,verbose=false)
    @test get_formatter(slv_ns).norm_style.output_style isa MScoreStyle

    # Opt-in SpatialKernelFormatter(;shared_coords=true) K=1 fast-path matches sample-by-sample
    shared_coords = repeat(rand(TData,2,N,1),1,1,K)
    var_feats = rand(TData,in_dims - 2,N,K)
    sp_in = vcat(shared_coords,var_feats)
    for cw in (true,false)
      kno_sp = KernelNeuralModel(
        gspace;
        hidden_layers=(8,8,8),
        channel_wise=cw,
        activation=tanh,
        formatter=SpatialKernelFormatter(;shared_coords=true)
      )
      ch_sp = build_model(kno_sp,(sp_in,dummy_out))
      ps_sp,st_sp = Lux.setup(Random.default_rng(),ch_sp)
      pred_batch,_ = ch_sp(sp_in,ps_sp,st_sp)
      for k in 1:K
        pred_k,_ = ch_sp(sp_in[:,:,k:k],ps_sp,st_sp)
        @test pred_batch[:,:,k:k] ≈ pred_k atol=1e-5
      end
    end

    # Default SpatialKernelFormatter() (shared_coords=false) handles sample-varying coordinates
    var_coords = rand(TData,2,N,K)
    sp_var_in = vcat(var_coords,var_feats)
    kno_sp_var = KernelNeuralModel(
      gspace;
      hidden_layers=(8,8,8),
      activation=tanh,
      formatter=SpatialKernelFormatter()
    )
    ch_sp_var = build_model(kno_sp_var,(sp_var_in,dummy_out))
    ps_sv,st_sv = Lux.setup(Random.default_rng(),ch_sp_var)
    pred_sv,_ = ch_sp_var(sp_var_in,ps_sv,st_sv)
    for k in 1:K
      pred_k,_ = ch_sp_var(sp_var_in[:,:,k:k],ps_sv,st_sv)
      @test pred_sv[:,:,k:k] ≈ pred_k atol=1e-5
    end

    # Residual flag controls skip connection on same-width nonlinear KernelNeuralLayers
    kno_res = KernelNeuralModel(gspace;hidden_layers=(8,8),residual=true,activation=tanh)
    kno_nores = KernelNeuralModel(gspace;hidden_layers=(8,8),residual=false,activation=tanh)
    ch_res = build_model(kno_res,(dummy_in,dummy_out))
    ch_nores = build_model(kno_nores,(dummy_in,dummy_out))
    @test ch_res.layers.layer_2.residual
    @test !ch_nores.layers.layer_2.residual
    ps_r,st_r = Lux.setup(Random.default_rng(),ch_res)
    pred_res,_ = ch_res(dummy_in,ps_r,st_r)
    pred_nores,_ = ch_nores(dummy_in,ps_r,st_r)
    @test !(pred_res ≈ pred_nores)

    # MultipleAttentionKernel concatenates heads along dims=1 and handles EdgeDataCache carriers
    gmeas_att = InducedMeasure(gspace)
    att_in_dim = 2 * in_dims + 1
    att_fmt = FeatureKernelFormatter()
    scores_att = (Lux.Dense(att_in_dim => 1),Lux.Dense(att_in_dim => 1))
    values_att = (Lux.Dense(att_in_dim => 4),Lux.Dense(att_in_dim => 4))
    agg_att = Lux.Dense(8 => 6)
    mak = MultipleAttentionKernel(scores_att,values_att,agg_att,gmeas_att,att_fmt)
    ps_att,st_att = Lux.setup(Random.default_rng(),mak)
    att_in = dummy_in[:,:,1:1]
    att_v = rand(TData,8,N,1)
    (_,z_att),_ = mak((att_in,att_v),ps_att,st_att)
    @test size(z_att) == (6,N,1)
    @test all(isfinite,z_att)
    # Passing an EdgeDataCache produced by a preceding KernelNeuralLayer must also work
    kl_cache = ch_res.layers.layer_2
    (cached_carrier,_),_ = kl_cache((att_in,att_v),ps_r.layer_2,st_r.layer_2)
    (_,z_att_cached),_ = mak((cached_carrier,att_v),ps_att,st_att)
    @test z_att_cached ≈ z_att

    # EdgeDataCache must not reuse edge tensors across layers with different measures
    meas_a = InducedMeasure(gspace;message_rule=PlainMP())
    meas_b = InducedMeasure(gspace;message_rule=AnchoredMP())
    ch_a = build_model(KernelNeuralModel(IntegralStyle(meas_a);hidden_layers=(8,8,8),activation=tanh),(dummy_in,dummy_out))
    ch_b = build_model(KernelNeuralModel(IntegralStyle(meas_b);hidden_layers=(8,8,8),activation=tanh),(dummy_in,dummy_out))
    ch_mix = Lux.Chain(ch_a.layers.layer_1,ch_a.layers.layer_2,ch_b.layers.layer_3,ch_a.layers.layer_4)
    ps_m,st_m = Lux.setup(Random.default_rng(),ch_mix)
    pred_mix,_ = ch_mix(dummy_in,ps_m,st_m)
    (_,v1),_ = ch_mix.layers.layer_1(dummy_in,ps_m.layer_1,st_m.layer_1)
    (_,v2),_ = ch_mix.layers.layer_2((dummy_in,v1),ps_m.layer_2,st_m.layer_2)
    (_,v3),_ = ch_mix.layers.layer_3((dummy_in,v2),ps_m.layer_3,st_m.layer_3)
    pred_ref,_ = ch_mix.layers.layer_4((dummy_in,v3),ps_m.layer_4,st_m.layer_4)
    @test pred_mix ≈ pred_ref atol=1e-5
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
    test_space = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags="boundary")
    trial_space = ParamTrialFESpace(test_space,gμ)
    feop = LinearParamFEOperator(rhs,stiffness,pspace,trial_space,test_space)
    fesolver = LinearFESolver(LUSolver())

    μ_train = realisation(pspace;nparams=8,sampling=:halton)
    uh_train = solve(fesolver,feop,μ_train)

    x_coords = get_coords(test_space)
    in_train = InputSpace(x_coords,FeatureBundle((aμ(μ_train),fμ(μ_train))))
    out_train = OutputSpace(uh_train)

    coords_backup = copy(in_train.coords)
    features_backup = copy(in_train.features)
    values_backup = copy(out_train.values)

    gspace = GraphInducedSpace(test_space;radius=0.35)
    gno = KernelNeuralModel(gspace;hidden_layers=(8,8),activation=tanh)
    solver = NeuralSolver(gno;epochs=5,batch_size=4,verbose=false)

    op = neural_operator(solver,in_train,out_train)

    # Training must not mutate the user's InputSpace or OutputSpace tensors in place
    @test in_train.coords ≈ coords_backup
    @test in_train.features ≈ features_backup
    @test out_train.values ≈ values_backup

    # Repeated online solve calls on the same InputSpace must be idempotent
    μ_test = realisation(pspace;nparams=3,sampling=:halton,start=9)
    in_test = InputSpace(x_coords,FeatureBundle((aμ(μ_test),fμ(μ_test))))
    test_coords_backup = copy(in_test.coords)
    test_features_backup = copy(in_test.features)

    pred1 = solve(solver,op,in_test)
    pred2 = solve(solver,op,in_test)

    @test in_test.coords ≈ test_coords_backup
    @test in_test.features ≈ test_features_backup
    @test pred1 ≈ pred2

    # Non-dividing mini-batch sizes (both >1 and ==1 batches after partial=false)
    # must compile and train cleanly on Reactant/XLA, and seed forwarding must be deterministic
    for bs in (3,5)
      slv_mb = NeuralSolver(gno;epochs=3,batch_size=bs,verbose=false)
      op_mb1 = neural_operator(slv_mb,in_train,out_train;seed=77)
      op_mb2 = neural_operator(slv_mb,in_train,out_train;seed=77)
      pred_mb1 = solve(slv_mb,op_mb1,in_test)
      pred_mb2 = solve(slv_mb,op_mb2,in_test)
      @test size(pred_mb1) == (1,length(x_coords),3)
      @test all(isfinite,pred_mb1)
      @test pred_mb1 ≈ pred_mb2
    end

    # Dropping tail samples with partial=false and shuffle=false must emit a warning
    slv_drop = NeuralSolver(gno;epochs=1,batch_size=3,verbose=false)
    dummy_dl_data = (rand(TData,2,4,8),rand(TData,1,4,8))
    @test_logs (:warn,r"Dropping") NonlinearROMs.get_data_loader(slv_drop,dummy_dl_data;shuffle=false)

    # End-to-end training with SpatialKernelFormatter, StateKernelFormatter, and non-uniform widths
    for (fmt,hlayers) in (
      (SpatialKernelFormatter(),(8,8)),
      (StateKernelFormatter(),(8,8)),
      (FeatureKernelFormatter(),(8,12,8))
    )
      gno_var = KernelNeuralModel(
        IntegralStyle(InducedMeasure(gspace;message_rule=PlainMP()));
        hidden_layers=hlayers,
        activation=tanh,
        formatter=fmt
      )
      slv_var = NeuralSolver(gno_var;epochs=3,batch_size=0,verbose=false)
      op_var = neural_operator(slv_var,in_train,out_train)
      pred_var = solve(slv_var,op_var,in_test)
      @test size(pred_var) == (1,length(x_coords),3)
      @test all(isfinite,pred_var)
    end

    # FunctionSpace + EnergyLoss(H1(),fspace) + to_element round-trip
    uh_test = solve(fesolver,feop,μ_test)
    fspace = FunctionSpace(test_space)
    h1_loss = EnergyLoss(H1(),fspace)
    fno = KernelNeuralModel(fspace;hidden_layers=(8,8),activation=tanh)
    fsolver = NeuralSolver(fno;epochs=15,batch_size=0,loss=h1_loss,verbose=false)

    fop = neural_operator(fsolver,in_train,out_train)
    pred_f = solve(fsolver,fop,in_test)
    @test size(pred_f) == (1,length(x_coords),3)
    @test all(isfinite,pred_f)

    pred_uh = to_element(OutputSpace(uh_test),pred_f)
    @test size(get_all_data(get_free_dof_values(pred_uh))) == size(get_all_data(get_free_dof_values(uh_test)))
  end

  @testset "MultipleMeasure, replace_space & Module Exports" begin
    # Verify all exported symbols actually exist in NonlinearROMs
    for sym in names(NonlinearROMs)
      @test isdefined(NonlinearROMs,sym)
    end

    # 1. Multi-field MultipleMeasure (Taylor-Hood Q2/Q1) + BlockNorm EnergyLoss
    pdomain = (1.0,4.0)
    pspace_mf = ParamSpace(pdomain)
    fesolver = LinearFESolver(LUSolver())

    model_mf = CartesianDiscreteModel((0,1,0,1),(3,3))
    Ω_mf = Triangulation(model_mf)
    dΩ_mf = Measure(Ω_mf,4)

    reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},2)
    reffe_p = ReferenceFE(lagrangian,Float64,1)
    Vu = TestFESpace(Ω_mf,reffe_u;conformity=:H1,dirichlet_tags="boundary")
    Vp = TestFESpace(Ω_mf,reffe_p;conformity=:H1)

    a_mf(μ) = x -> μ[1]
    a_mfμ(μ) = parameterise(a_mf,μ)
    g_mf(μ) = x -> VectorValue(0.0,0.0)
    g_mfμ(μ) = parameterise(g_mf,μ)

    Uu = ParamTrialFESpace(Vu,g_mfμ)
    Up = ParamTrialFESpace(Vp)
    V_mf = MultiFieldFESpace([Vu,Vp])
    U_mf = MultiFieldFESpace([Uu,Up])

    stiff_mf(μ,(u,p),(v,q)) = ∫(a_mfμ(μ)*(∇(v)⊙∇(u)) - p*(∇⋅v) + q*(∇⋅u))dΩ_mf
    rhs_mf(μ,(v,q)) = ∫(VectorValue(1.0,0.0)⋅v)dΩ_mf
    feop_mf = LinearParamFEOperator(rhs_mf,stiff_mf,pspace_mf,U_mf,V_mf)
    μ_mf = realisation(pspace_mf;nparams=3)
    uh_mf = solve(fesolver,feop_mf,μ_mf)

    in_mf = InputSpace(V_mf,FeatureBundle((a_mfμ(μ_mf),)))
    out_mf = OutputSpace(uh_mf)
    fspace_mf = FunctionSpace(V_mf)
    gspace_mf = GraphInducedSpace(V_mf;radius=0.5)
    gmeas_mf = InducedMeasure(gspace_mf)
    @test gmeas_mf isa MultipleMeasure

    N_mf = num_coords(gspace_mf)
    E_mf = num_edges(gmeas_mf)
    vals_mf = rand(TData,4,N_mf,2)
    g_mf_data = gather_data(vals_mf,gmeas_mf)
    @test size(g_mf_data) == (2 * 4 + 1,E_mf,2)
    @test size(scatter_data(g_mf_data,gmeas_mf)) == (2 * 4 + 1,N_mf,2)
    int_mf = integration(g_mf_data,gmeas_mf)
    @test size(int_mf) == (2 * 4 + 1,N_mf,2)
    @test all(isfinite,int_mf)

    # Multi-field BlockNorm EnergyLoss
    blk_loss = EnergyLoss(BlockNorm((H1(),L2())),fspace_mf)
    u_mf_rand = rand(TData,3,N_mf,2)
    u_mf_zero = zeros(TData,3,N_mf,2)
    @test blk_loss(u_mf_rand,u_mf_rand) ≈ 0.0f0 atol=1e-6
    @test blk_loss(u_mf_rand,u_mf_zero) > 0.0f0
    X_blk = assemble_norm(BlockNorm((H1(),L2())),fspace_mf)
    expected_blk = mean(begin
      ek = vcat(u_mf_rand[1,:,k],u_mf_rand[2,:,k],u_mf_rand[3,:,k])
      dot(ek,X_blk * ek)
    end for k in 1:2)
    @test isapprox(blk_loss(u_mf_rand,u_mf_zero),expected_blk;rtol=1e-4)

    gno_mf = KernelNeuralModel(gspace_mf;hidden_layers=(8,8),activation=tanh)
    slv_mf = NeuralSolver(gno_mf;epochs=2,batch_size=0,loss=blk_loss,verbose=false)
    op_mf = neural_operator(slv_mf,in_mf,out_mf)
    pred_mf = solve(slv_mf,op_mf,in_mf)
    @test size(pred_mf) == (3,N_mf,3)
    @test all(isfinite,pred_mf)

    # Multi-field MultipleMeasure on FunctionSpace (FESpaceMeasure local blocks)
    fmeas_mf = InducedMeasure(fspace_mf)
    @test fmeas_mf isa MultipleMeasure
    E_fmf = num_edges(fmeas_mf)
    g_fmf_data = gather_data(vals_mf,fmeas_mf)
    @test size(g_fmf_data) == (2 * 4 + 1,E_fmf,2)
    int_fmf = integration(g_fmf_data,fmeas_mf)
    @test size(int_fmf) == (2 * 4 + 1,N_mf,2)
    @test all(isfinite,int_fmf)
    fno_mf = KernelNeuralModel(fspace_mf;hidden_layers=(8,8),activation=tanh)
    fslv_mf = NeuralSolver(fno_mf;epochs=2,batch_size=0,loss=blk_loss,verbose=false)
    fop_mf = neural_operator(fslv_mf,in_mf,out_mf)
    fpred_mf = solve(fslv_mf,fop_mf,in_mf)
    @test size(fpred_mf) == (3,N_mf,3)
    @test all(isfinite,fpred_mf)

    # 2. Multi-PDE MultipleMeasure + replace_space onto a single new mesh
    reffe_s = ReferenceFE(lagrangian,Float64,1)
    g_s(μ) = x -> 0.0
    g_sμ(μ) = parameterise(g_s,μ)

    function solve_poisson_mpde(n,np,start)
      m = CartesianDiscreteModel((0,1,0,1),(n,n))
      t = Triangulation(m)
      dm = Measure(t,2)
      V = TestFESpace(t,reffe_s;conformity=:H1,dirichlet_tags="boundary")
      U = ParamTrialFESpace(V,g_sμ)
      op = LinearParamFEOperator((μ,v) -> ∫(a_mfμ(μ)*v)dm,(μ,u,v) -> ∫(∇(v)⋅∇(u))dm,pspace_mf,U,V)
      μ = realisation(pspace_mf;nparams=np,start=start)
      uh = solve(fesolver,op,μ)
      return V,InputSpace(V,FeatureBundle((a_mfμ(μ),))),uh
    end

    V1,in1,uh1 = solve_poisson_mpde(3,2,1)
    V2,in2,uh2 = solve_poisson_mpde(4,3,3)
    in_mpde = InputSpace(in1,in2)
    out_mpde = OutputSpace(uh1,uh2)
    gspace_mpde = GraphInducedSpace(out_mpde;radius=0.5)
    @test InducedMeasure(gspace_mpde) isa MultipleMeasure

    gno_mpde = KernelNeuralModel(gspace_mpde;hidden_layers=(8,8),activation=tanh)
    slv_mpde = NeuralSolver(gno_mpde;epochs=2,batch_size=0,verbose=false)
    op_mpde = neural_operator(slv_mpde,in_mpde,out_mpde)
    pred_mpde = solve(slv_mpde,op_mpde,in_mpde)
    @test size(pred_mpde) == (1,num_coords(out_mpde),5)
    @test all(isfinite,pred_mpde)

    # Transfer multi-PDE trained operator onto a single-PDE graph space
    op_single = replace_space(op_mpde,GraphInducedSpace(V2;radius=0.45))
    pred_single = solve(slv_mpde,op_single,in2)
    @test size(pred_single) == (1,length(get_coords(V2)),3)
    @test all(isfinite,pred_single)

    # 3. replace_space across meshes and MPRule preservation
    model_c = CartesianDiscreteModel((0,1,0,1),(5,5))
    model_f = CartesianDiscreteModel((0,1,0,1),(8,8))
    reffe = ReferenceFE(lagrangian,Float64,1)
    Vc = TestFESpace(model_c,reffe;conformity=:H1,dirichlet_tags="boundary")
    Vf = TestFESpace(model_f,reffe;conformity=:H1,dirichlet_tags="boundary")

    pspace = ParamSpace((1.0,3.0))
    μ = realisation(pspace;nparams=4)
    fb = FeatureBundle((parameterise(μ -> x -> μ[1],μ),))

    gspace_c = GraphInducedSpace(Vc;radius=0.4)
    gspace_f = GraphInducedSpace(Vf;radius=0.25)

    gno = KernelNeuralModel(gspace_c;hidden_layers=(8,8),activation=tanh)
    solver = NeuralSolver(gno;epochs=3,verbose=false)
    in_c = InputSpace(Vc,fb)
    out_c = OutputSpace(Vc,rand(TData,1,length(get_coords(Vc)),4))

    op_c = neural_operator(solver,in_c,out_c)
    op_f = replace_space(op_c,gspace_f)
    pred_f = solve(solver,op_f,InputSpace(Vf,fb))
    @test size(pred_f) == (1,length(get_coords(Vf)),4)
    @test all(isfinite,pred_f)

    # replace_space preserves all message passing rules and remains runnable on the new mesh
    for rule in (DifferenceMP(),PlainMP(),CompositeMP(),AnchoredMP())
      meas_r = InducedMeasure(gspace_c;message_rule=rule)
      gno_r = KernelNeuralModel(IntegralStyle(meas_r);hidden_layers=(8,8),activation=tanh)
      slv_r = NeuralSolver(gno_r;epochs=2,verbose=false)
      op_r = neural_operator(slv_r,in_c,out_c)
      op_r_f = replace_space(op_r,gspace_f)
      @test get_message_rule(op_r_f.chain.layers.layer_2.kernel.measure) isa typeof(rule)
      pred_r_f = solve(slv_r,op_r_f,InputSpace(Vf,fb))
      @test size(pred_r_f) == (1,length(get_coords(Vf)),4)
      @test all(isfinite,pred_r_f)
    end
  end
end