using DrWatson
using Gridap
using GridapEmbedded
using GridapROMs
using NonlinearROMs
import Gridap.ODEs: get_test

L = 2
W = 2
N = 20
γd = 10.0
hd = max(L,W)/N

domain = (0,L,0,W)
partition = (N,N)
bgmodel = CartesianDiscreteModel(domain,partition)

order = 1
degree = 2*order

pdomain = (1,10,1,10,0,1)
pspace = ParamSpace(pdomain)

reffe = ReferenceFE(lagrangian,Float64,order)

a(μ) = x -> μ[1]
aμ(μ) = parameterise(a,μ)

f(μ) = x -> μ[2]
fμ(μ) = parameterise(f,μ)

g(μ) = x -> 0.0
gμ(μ) = parameterise(g,μ)

function pde(R,x0)
  geo = !disk(R,x0=x0)
  cutgeo = cut(bgmodel,geo)

  Ωact = Triangulation(cutgeo,ACTIVE)
  Ω = Triangulation(cutgeo,PHYSICAL)
  Γ = EmbeddedBoundary(cutgeo)

  dΩ = Measure(Ω,degree)
  dΓ = Measure(Γ,degree)

  n_Γ = get_normal_vector(Γ)
  strategy = AggregateAllCutCells()
  aggregates = aggregate(strategy,cutgeo)

  stiffness(μ,u,v) = ∫(aμ(μ)*∇(v)⋅∇(u))dΩ + ∫( (γd/hd)*v*u - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ
  rhs(μ,v) = ∫(fμ(μ)*v)dΩ

  testact = TestFESpace(Ωact,reffe,conformity=:H1,dirichlet_tags=[1,3,7])
  test = AgFEMSpace(testact,aggregates)
  trial = ParamTrialFESpace(test,gμ)
  feop = LinearParamFEOperator(rhs,stiffness,pspace,trial,test)

  return feop
end

fesolver = LinearFESolver(LUSolver())

dir = datadir("moving_poisson")

features(μ) = FeatureBundle((aμ(μ),fμ(μ),gμ(μ)))

# PDE 1
feop1 = pde(0.1,Point(0.5,1.5))
test1 = get_test(feop1)
μ_train1 = realisation(pspace;nparams=25)
uhμ_train1 = solve(fesolver,feop1,μ_train1)

# PDE 2
feop2 = pde(0.2,Point(1.5,0.5))
test2 = get_test(feop2)
μ_train2 = realisation(pspace;nparams=25)
uhμ_train2 = solve(fesolver,feop2,μ_train2)

# PDE 3
feop = pde(0.3,Point(1.0,1.0))
test = get_test(feop)
μ = realisation(pspace;nparams=1)
uhμ = solve(fesolver,feop,μ)

# training data of the two PDEs stacked sample-wise on the union of their points
input_space = InputSpace(InputSpace(test1,features(μ_train1)),InputSpace(test2,features(μ_train2)))
output_space = OutputSpace(uhμ_train1,uhμ_train2)

# 1) GNO
gspace = GraphInducedSpace(output_space;radius=hd)
gmodel = KernelNeuralModel(gspace;hidden_layers=(64,64,64),activation=tanh)
solver = NeuralSolver(gmodel;epochs=1000,loss=EnergyLoss(H1(),fspace),print_every=50)
op = neural_operator(solver,input_space,output_space)
# evaluation on a new mesh: same trained network, kernels integrated on the new graph
gspace_test = GraphInducedSpace(test;radius=hd)
op_test = replace_space(op,gspace_test)
eval_input_space = InputSpace(test,features(μ))
pred_uμ = solve(solver,op_test,eval_input_space)
pred_uhμ = to_element(OutputSpace(uhμ),pred_uμ)
visualise(dir,uhμ,pred_uhμ;variable_name="uh_gno")

# 2) FNO
kmodel = KernelNeuralModel(FourierStyle(truncation=12);hidden_layers=(64,64,64,64),activation=tanh)
solver = NeuralSolver(kmodel;epochs=1000,print_every=50)
op = neural_operator(solver,input_space,output_space)
# no new mesh: integral is in the Fourier domain
eval_input_space = InputSpace(test,features(μ))
pred_uμ = solve(solver,op,eval_input_space)
pred_uhμ = to_element(OutputSpace(uhμ),pred_uμ)
visualise(dir,uhμ,pred_uhμ;variable_name="uh_fno")