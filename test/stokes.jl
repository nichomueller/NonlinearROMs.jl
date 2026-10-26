using Gridap 
using GridapROMs
using GridapROMs.RBSteady
using NonlinearROMs

pdomain = (1,10,-1,5,1,2)
pspace = ParamSpace(pdomain)

domain = (0,1,0,1)
partition = (8,8)
model = CartesianDiscreteModel(domain,partition)

order = 2
degree = 2*order

Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

a(μ) = x -> μ[1]*exp(-x[1])
aμ(μ) = parameterise(a,μ)

f(x) = VectorValue(0.0,0.0)

g(μ) = x -> VectorValue(-(μ[2]*x[2]+μ[3])*x[2]*(1.0-x[2]),0.0)*(x[1]==0.0)
gμ(μ) = parameterise(g,μ)

stiffness(μ,(u,p),(v,q)) = ∫(aμ(μ)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
rhs(μ,(v,q)) = ∫(f⋅v)dΩ

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_u = TestFESpace(Ω,reffe_u;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7])
test_p = TestFESpace(Ω,reffe_p;conformity=:H1)
trial_u = ParamTrialFESpace(test_u,gμ)
trial_p = ParamTrialFESpace(test_p)
test = MultiFieldFESpace([test_u,test_p])
trial = MultiFieldFESpace([trial_u,trial_p])

feop = LinearParamFEOperator(rhs,stiffness,pspace,trial,test)
fesolver = LinearFESolver(LUSolver())

μ = realisation(pspace;nparams=50)
uhμ = solve(fesolver,feop,μ)

using GridapROMs.ParamDataStructures
using Gridap.Algebra

features = FeatureBundle((aμ(μ),gμ(μ)))
fspace = FunctionSpace(test)
gspace = GraphInducedSpace(fspace;radius=0.3)

input_space = InputSpace(fspace,features)
output_space = OutputSpace(uhμ)

model = KernelNeuralModel(gspace;hidden_layers=(16,16),activation=tanh)
X = assemble_matrix(((u,p),(v,q)) -> ∫(∇(v)⊙∇(u) + q*p)dΩ,trial,test)
solver = NeuralSolver(fesolver,model;epochs=200,loss=EnergyLoss(X),print_every=50)

op = neural_operator(solver,input_space,output_space)
pred = solve(solver,op,input_space)
pred_uh = Algebra.solve(solver,op,input_space,output_space)