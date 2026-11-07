using Test
using LinearAlgebra
using Statistics
using Gridap
using Gridap.FESpaces
using GridapROMs
using GridapROMs.ParamDataStructures
using GridapROMs.ParamFESpaces
using GridapROMs.ParamSteady
using GridapSolvers
using GridapSolvers.LinearSolvers
using NonlinearROMs

using NonlinearROMs: TData,MScore,ZScore,Normalisation,normalise!,denormalise!
using NonlinearROMs: fetch_data,fetch_stats,merge_data,num_coords,num_samples

@testset "DataInterface & Normalisation" begin

  @testset "ZScore & MScore Normalisation" begin
    D,N,K = 3,25,10
    data = zeros(TData,D,N,K)

    # Channel 1: Spatial coordinates (vary across nodes N, constant across samples K)
    x_nodes = range(0.0f0,1.0f0,length=N)
    for k in 1:K,i in 1:N
      data[1,i,k] = x_nodes[i]
    end

    # Channel 2: Parameter-independent spatial source term f(x) = sin(π*x)
    for k in 1:K,i in 1:N
      data[2,i,k] = sin(Float32(π) * x_nodes[i])
    end

    # Channel 3: Global constant field across all nodes and samples (e.g. f(x) = 5.0)
    fill!(view(data,3,:,:),5.0f0)

    data_orig = copy(data)
    stats = ZScore(data; normalise=true)

    # Spatial profiles must be preserved with unit standard deviation across (N, K),
    # not wiped out to numerical zero by per-node statistics
    @test norm(view(data,1,:,1)) > 1.0f0
    @test norm(view(data,2,:,1)) > 1.0f0
    @test std(view(data,1,:,:)) ≈ 1.0f0 atol=1e-4
    @test std(view(data,2,:,:)) ≈ 1.0f0 atol=1e-4

    # Constant channels (zero variance) must not be zeroed out or produce NaNs
    @test all(x -> isapprox(x,5.0f0; atol=1e-5),view(data,3,:,:))
    @test !any(isnan,data)

    # Discretization invariance: statistics must have shape (D, 1, 1) so they can
    # be applied to a mesh with a different number of evaluation points N_test != N
    @test size(stats.μ) == (D,1,1)
    @test size(stats.σ) == (D,1,1)

    N_test = 37
    data_new_mesh = rand(TData,D,N_test,4)
    original = copy(data_new_mesh)

    normalise!(data_new_mesh,stats)
    denormalise!(data_new_mesh,stats)

    @test data_new_mesh ≈ original atol=1e-5

    # Exact round-trip recovery
    denormalise!(data,stats)
    @test data ≈ data_orig atol=1e-5

    # MScore round-trip recovery
    out_data = randn(TData,2,N,K) .* 10.0f0
    out_orig = copy(out_data)
    mstats = MScore(out_data; normalise=true)
    @test maximum(abs,out_data) ≈ 1.0f0 atol=1e-5
    denormalise!(out_data,mstats)
    @test out_data ≈ out_orig atol=1e-5
  end

  @testset "DataFormatter & Kernel Formulations" begin
    coord_dims = 2
    feat_dims = 4
    fixed_dims = coord_dims + feat_dims
    latent_dims = 8
    N,K = 15,5

    coords = rand(TData,coord_dims,N,K)
    features = rand(TData,feat_dims,N,K)
    fixed_data = vcat(coords,features)
    v_state = rand(TData,latent_dims,N,K)

    # Spatial subsampling via Sampler
    sub_ids = 1:2:N
    fmt_sub = DataFormatter(sampler=Sampler(sub_ids))
    fetched = fetch_data(fmt_sub,(coords=coords,features=features))
    @test size(fetched.coords) == (coord_dims,length(sub_ids),K)
    @test size(fetched.features) == (feat_dims,length(sub_ids),K)

    # Kovachki et al. Eq. (7): κ(x, y) -> coordinates only
    fmt_spatial = SpatialKernelFormatter()
    m_spatial = merge_data(fmt_spatial,fixed_data,v_state,coord_dims)
    @test size(m_spatial) == (coord_dims,N,K)
    @test m_spatial == view(fixed_data,1:coord_dims,:,:)

    # Kovachki et al. Eq. (8): κ(x, y, a(x), a(y)) -> coordinates and PDE features
    fmt_feature = FeatureKernelFormatter()
    m_feature = merge_data(fmt_feature,fixed_data,v_state,coord_dims)
    @test size(m_feature) == (fixed_dims,N,K)
    @test m_feature == fixed_data

    # Kovachki et al. Eq. (9): κ(x, y, v(x), v(y)) -> coordinates and latent state v
    fmt_state = StateKernelFormatter()
    m_state = merge_data(fmt_state,fixed_data,v_state,coord_dims)
    @test size(m_state) == (coord_dims + latent_dims,N,K)
    @test m_state[1:coord_dims,:,:] == view(fixed_data,1:coord_dims,:,:)
    @test m_state[(coord_dims+1):end,:,:] == v_state

    # Full formulation: κ(x, y, a(x), a(y), v(x), v(y))
    fmt_full = FullKernelFormatter()
    m_full = merge_data(fmt_full,fixed_data,v_state,coord_dims)
    @test size(m_full) == (fixed_dims + latent_dims,N,K)
  end

  @testset "InputSpace, OutputSpace & FEFunction Reconstruction" begin
    pdomain = (1.0,5.0)
    pspace = ParamSpace(pdomain)
    model = CartesianDiscreteModel((0,1,0,1),(4,4))
    Ω = Triangulation(model)
    dΩ = Measure(Ω,2)

    a(μ) = x -> μ[1] * (1.0 + x[1])
    aμ(μ) = parameterise(a,μ)
    f(μ) = x -> 2.0 * sin(π * x[1]) * sin(π * x[2])
    fμ(μ) = parameterise(f,μ)
    g(μ) = x -> 0.0
    gμ(μ) = parameterise(g,μ)

    stiffness(μ,u,v) = ∫(aμ(μ) * ∇(v) ⋅ ∇(u))dΩ
    rhs(μ,v) = ∫(fμ(μ) * v)dΩ

    reffe = ReferenceFE(lagrangian,Float64,1)
    V = TestFESpace(Ω,reffe; conformity=:H1,dirichlet_tags="boundary")
    U = ParamTrialFESpace(V,gμ)
    feop = LinearParamFEOperator(rhs,stiffness,pspace,U,V)
    fesolver = LinearFESolver(LUSolver())

    μ = realisation(pspace; nparams=4)
    uh = solve(fesolver,feop,μ)

    fb = FeatureBundle((aμ(μ),fμ(μ)))
    # Test constructor accepting FESpace directly as first argument
    in_space = InputSpace(V,fb)
    out_space = OutputSpace(uh)

    N = length(get_coords(V))

    @test size(get_coords(in_space)) == (2,N,4)
    @test size(get_features(in_space)) == (2,N,4)
    @test size(out_space.values) == (1,N,4)

    # Round-trip from OutputSpace tensor back to ParamFEFunction
    uh_reconstructed = to_element(out_space,out_space.values)
    @test get_all_data(get_free_dof_values(uh_reconstructed)) ≈ get_all_data(get_free_dof_values(uh)) atol=1e-5
  end

end
