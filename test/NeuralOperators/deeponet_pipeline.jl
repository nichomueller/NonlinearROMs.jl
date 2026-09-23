using Test
using Gridap
using GridapROMs
using GridapROMs.RBSteady
using GridapROMs.ParamDataStructures
using GridapROMs.ParamSteady
using GridapROMs.ParamODEs
using GridapROMs.Utils
using GridapROMs.DofMaps
using NonlinearROMs
using LinearAlgebra

# Mock quantities
const MockModel = CartesianDiscreteModel((0.0,1.0),(2,))
const MockReffe = ReferenceFE(lagrangian,Float64,1)

struct MockSteadyOpDON <: ParamOperator{LinearParamEq,JointDomains} end
Gridap.FESpaces.get_test(::MockSteadyOpDON) = LexicographicFESpace(MockModel,MockReffe)

struct MockTransientOpDON <: ParamOperator{LinearParamODE,JointDomains} end
Gridap.FESpaces.get_test(::MockTransientOpDON) = LexicographicFESpace(MockModel,MockReffe)

@testset "DeepONet Steady Integration Pipeline" begin
  feop = MockSteadyOpDON()

  # Data generation (4 samples,2 parameters,3 DoF)
  n_samples = 4
  n_params = 2
  N_dofs = 3
  
  param_values = [rand(Float32,n_params) for _ in 1:n_samples]
  r = Realisation(param_values)
  
  u_data = rand(Float64,N_dofs,n_samples)
  snaps = Snapshots(ConsecutiveParamArray(u_data),VectorDofMap(N_dofs),r)

  # Strategy and Solver
  strategy = NeuralStrategy(
    DeepONet(2,1;width=8,depth=1),
    epochs = 2,
    batch_size = 2,
    lr_scheduler = CosineAnnealing(2,lr_max=0.01f0,lr_min=0.001f0),
    verbose=false
  )
  reduction = DeepONetReduction(strategy)
  solver = NeuralSolver(LUSolver(),reduction)

  # Offline Phase
  neural_op = reduced_operator(solver,feop,snaps)
  
  @test neural_op isa NeuralOperator
  @test neural_op.metadata.dmax > 0
  
  # Online Phase
  r_test = Realisation([[0.5f0,0.5f0]])
  x_hat,stats = solve(solver,neural_op,r_test)
  
  @test x_hat isa Snapshots
  @test size(get_all_data(x_hat)) == (3,1) # 3 DoF,1 Sample
  @test stats.name == "DeepONet Inference"
end

@testset "DeepONet Transient Integration" begin
  feop = MockTransientOpDON()

  n_samples,n_params,N_dofs,N_time = 2,2,3,2
  
  # TransientSnapshots creation
  params = [rand(Float32,n_params) for _ in 1:n_samples]
  times = [0.0,1.0]
  r = TransientRealisation(Realisation(params),times,0.0)
  
  u_data = rand(Float64,N_dofs,n_samples,N_time)
  # GridapROMs' transient Snapshots constructor expects the underlying ConsecutiveParamArray
  # to wrap a flat (N_dofs,n_samples*N_time) array (param varying fastest); a plain `reshape`
  # of the (N_dofs,n_samples,N_time) array achieves exactly that column order.
  snaps = Snapshots(ConsecutiveParamArray(reshape(u_data,N_dofs,n_samples*N_time)),VectorDofMap(N_dofs),r)

  # Solver Transient (trunk input is 1D coords + time = 2)
  strategy = NeuralStrategy(DeepONet(2,2;width=8,depth=1),epochs=1,verbose=false)
  reduction = DeepONetReduction(strategy)
  solver = NeuralSolver(LUSolver(),reduction)

  neural_op = reduced_operator(solver,feop,snaps)
  
  # Inference
  x_hat,stats = solve(solver,neural_op,r)
  
  @test x_hat isa TransientSnapshots
  @test size(get_all_data(x_hat)) == (3,2,2)
  @test stats.name == "DeepONet Transient Inference"
end

@testset "Fine-Tuning Integration (DeepONet)" begin
  feop = MockSteadyOpDON()

  u_data = rand(Float64,3,4)
  snaps = Snapshots(ConsecutiveParamArray(u_data),VectorDofMap(3),Realisation([rand(2) for _ in 1:4]))
  
  # Setup solver
  strategy = NeuralStrategy(DeepONet(2,1;width=8,depth=1),epochs=1,verbose=false)
  reduction = DeepONetReduction(strategy)
  solver = NeuralSolver(LUSolver(),reduction)

  # First training
  pretrained_op = reduced_operator(solver,feop,snaps)
  
  # Fine-tuning
  new_op = reduced_operator(solver,feop,snaps,pretrained_op;update_stats=true)
  
  @test new_op isa NeuralOperator
  @test new_op.model.chain === pretrained_op.model.chain
end

@testset "Error Handling and Edge Cases (DeepONet)" begin
  @testset "Fine-Tuning Branch Dimension Mismatch" begin
    feop = MockSteadyOpDON() 
    
    # Base training with 2 sensors/parameters
    snaps_base = Snapshots(ConsecutiveParamArray(rand(Float64,3,2)),VectorDofMap(3),Realisation([rand(Float32,2) for _ in 1:2]))
    strategy = NeuralStrategy(DeepONet(2,1;width=4,depth=1),epochs=1,verbose=false)
    solver = NeuralSolver(LUSolver(),DeepONetReduction(strategy))
    
    pretrained_op = reduced_operator(solver,feop,snaps_base)

    # Fine-tuning with 3 sensors/parameters
    snaps_mismatch = Snapshots(ConsecutiveParamArray(rand(Float64,3,2)),VectorDofMap(3),Realisation([rand(Float32,3) for _ in 1:2]))
    
    # AssertionError
    @test_throws AssertionError reduced_operator(solver,feop,snaps_mismatch,pretrained_op)
  end
end