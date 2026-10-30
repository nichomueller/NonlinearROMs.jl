using DrWatson
using Gridap
using GridapEmbedded
using GridapROMs
using NonlinearROMs

const L = 2
const W = 2
const n = 20
const γd = 10.0
const hd = max(L,W)/n

domain = (0,L,0,W)
partition = (n,n)
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

g(μ) = x -> μ[3]
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

  return (test,trial,feop)
end

fesolver = LinearFESolver(LUSolver())

features(μ) = FeatureBundle((aμ(μ),fμ(μ),gμ(μ)))

# PDE 1
test1,trial1,feop1 = pde(0.1,Point(0.5,1.5))
μ_train1 = realisation(pspace;nparams=25)
uhμ_train1 = solve(fesolver,feop1,μ_train1)

# PDE 2
test2,trial2,feop2 = pde(0.2,Point(1.5,0.5))
μ_train2 = realisation(pspace;nparams=25)
uhμ_train2 = solve(fesolver,feop2,μ_train2)

# PDE 3
test,trial,feop = pde(0.3,Point(1.0,1.0))
μ = realisation(pspace;nparams=1)
uhμ = solve(fesolver,feop,μ)

# on variable meshes, we need a graph-based approach

gspace = MultiGraphInducedSpace(test1,test2;radius=2*hd)
gmodel = KernelNeuralModel(gspace;hidden_layers=(16,16),activation=tanh)
solver = NeuralSolver(fesolver,gmodel;epochs=200,print_every=50)
input_space = InputSpace(get_coords(gspace),features(μ_train1),features(μ_train2))
output_space = OutputSpace(uhμ_train1,uhμ_train2)

op = neural_operator(solver,input_space,output_space)
pred_uμ = solve(solver,op,input_space(μ))
pred_uhμ = to_element(OutputSpace(uhμ),pred_uμ)
visualise(dir,uhμ,pred_uhμ;variable_name="uh_gno")

using Gridap.FESpaces
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.CellData