using DrWatson
using Gridap 
using GridapROMs
using GridapROMs.RBSteady
using NonlinearROMs

pdomain = (1,10,1,10,1,10)
pspace = ParamSpace(pdomain)

N = 20
domain = (0,1,0,1)
partition = (N,N)
model = CartesianDiscreteModel(domain,partition)

order = 1
degree = 2*order

Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=[8])
dΓn = Measure(Γn,degree)

a(μ) = x -> exp(-x[1]/sum(μ))
aμ(μ) = parameterise(a,μ)

f(μ) = x -> 1.
fμ(μ) = parameterise(f,μ)

g(μ) = x -> μ[1]*exp(-x[1]/μ[2])
gμ(μ) = parameterise(g,μ)

h(μ) = x -> abs(cos(μ[3]*x[2]))
hμ(μ) = parameterise(h,μ)

stiffness(μ,u,v) = ∫(aμ(μ)*∇(v)⋅∇(u))dΩ
rhs(μ,v) = ∫(fμ(μ)*v)dΩ + ∫(hμ(μ)*v)dΓn

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
trial = ParamTrialFESpace(test,gμ)

feop = LinearParamFEOperator(rhs,stiffness,pspace,trial,test)
fesolver = LinearFESolver(LUSolver())

# FE quantities (snapshots)

μ_train = realisation(pspace;nparams=50)
uhμ_train = solve(fesolver,feop,μ_train)

μ = realisation(pspace;nparams=1)
uhμ = solve(fesolver,feop,μ)

# NO setup

dir = datadir("poisson")

x = get_coords(test)
features(μ) = FeatureBundle((aμ(μ),fμ(μ),gμ(μ),hμ(μ)))
input_space(μ) = InputSpace(x,features(μ))
output_space = OutputSpace(uhμ_train)

for epochs in (200,500,1000)
  for lsize in (8,16,64)
    for nsize in (2,3)
      hidden_layers = ntuple(i->lsize,nsize)
      case = "e=$(epochs)_l=$(lsize)_n=$(nsize)"
      # test 1: Gridap-based spaces

      fspace = FunctionSpace(test)
      fmodel = KernelNeuralModel(fspace;hidden_layers,activation=tanh)
      solver = NeuralSolver(fmodel;epochs,loss=EnergyLoss(H1(),fspace),print_every=50)
      op = neural_operator(solver,input_space(μ_train),output_space)
      pred_uμ = solve(solver,op,input_space(μ))
      pred_uhμ = to_element(OutputSpace(uhμ),pred_uμ)
      visualise(dir,uhμ,pred_uhμ;variable_name="uh_gridap_$(case)")

      # test 2: GNOs

      mesh_size = 1/N
      gspace = GraphInducedSpace(test;radius=2*mesh_size)
      gmodel = KernelNeuralModel(gspace;hidden_layers,activation=tanh)
      solver = NeuralSolver(gmodel;epochs,loss=EnergyLoss(H1(),fspace),print_every=50)
      op = neural_operator(solver,input_space(μ_train),output_space)
      pred_uμ = solve(solver,op,input_space(μ))
      pred_uhμ = to_element(OutputSpace(uhμ),pred_uμ)
      visualise(dir,uhμ,pred_uhμ;variable_name="uh_gno_$(case)")

      # test 3: FNOs

      kmodel = KernelNeuralModel(FourierStyle(truncation=12);hidden_layers,activation=tanh)
      solver = NeuralSolver(kmodel;epochs,loss=EnergyLoss(H1(),fspace),print_every=50)
      op = neural_operator(solver,input_space(μ_train),output_space)
      pred_uμ = solve(solver,op,input_space(μ))
      pred_uhμ = to_element(OutputSpace(uhμ),pred_uμ)
      visualise(dir,uhμ,pred_uhμ;variable_name="uh_fno_$(case)")
    end
  end
end

# # test 1: Gridap-based spaces

# fspace = FunctionSpace(test)
# fmodel = KernelNeuralModel(fspace;hidden_layers=(64,64,64),activation=tanh)
# solver = NeuralSolver(fmodel;epochs=1000,loss=EnergyLoss(H1(),fspace),print_every=50)
# op = neural_operator(solver,input_space(μ_train),output_space)
# pred_uμ = solve(solver,op,input_space(μ))
# pred_uhμ = to_element(OutputSpace(uhμ),pred_uμ)
# visualise(dir,uhμ,pred_uhμ;variable_name="uh_gridap")

# # test 2: GNOs

# mesh_size = 1/5
# gspace = GraphInducedSpace(test;radius=2*mesh_size)
# gmodel = KernelNeuralModel(gspace;hidden_layers=(64,64,64),activation=tanh)
# solver = NeuralSolver(gmodel;epochs=1000,loss=EnergyLoss(H1(),fspace),print_every=50)
# op = neural_operator(solver,input_space(μ_train),output_space)
# pred_uμ = solve(solver,op,input_space(μ))
# pred_uhμ = to_element(OutputSpace(uhμ),pred_uμ)
# visualise(dir,uhμ,pred_uhμ;variable_name="uh_gno")

# # test 3: FNOs

# kmodel = KernelNeuralModel(FourierStyle(truncation=12);hidden_layers=(64,64,64),activation=tanh)
# solver = NeuralSolver(kmodel;epochs=1000,loss=EnergyLoss(H1(),fspace),print_every=50)
# op = neural_operator(solver,input_space(μ_train),output_space)
# pred_uμ = solve(solver,op,input_space(μ))
# pred_uhμ = to_element(OutputSpace(uhμ),pred_uμ)
# visualise(dir,uhμ,pred_uhμ;variable_name="uh_fno")