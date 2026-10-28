using DrWatson
using Gridap
using GridapROMs
using GridapROMs.RBSteady
using Lux
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

# FE quantities (snapshots)

μ_train = realisation(pspace;nparams=50)
uhμ_train = solve(fesolver,feop,μ_train)

μ = realisation(pspace;nparams=1)
uhμ = solve(fesolver,feop,μ)

# NO setup

dir = datadir("stokes")

x = get_coords(test)
features(μ) = FeatureBundle((aμ(μ),gμ(μ)))
input_space(μ) = InputSpace(x,features(μ))
output_space = OutputSpace(uhμ_train)

# EnergyLoss's matrix multiplies along the model's *point* axis, shared across
# every channel -- it only happens to line up with a plain `assemble_matrix`
# Gram matrix for a scalar, order-1 field, where dof count == point count (see
# poisson.jl). Here velocity is vector-valued and the two fields have
# different polynomial orders, so dof count != point count and a dof-level
# Gram matrix can't be reused as-is; a plain MSE loss is used instead.
loss = Lux.MSELoss()

# test 1: Gridap-based spaces

fspace = FunctionSpace(test)
fmodel = KernelNeuralModel(fspace;hidden_layers=(16,16),activation=tanh)
solver = NeuralSolver(fesolver,fmodel;epochs=200,loss=loss,print_every=50)
op = neural_operator(solver,input_space(μ_train),output_space)
pred_uμ = solve(solver,op,input_space(μ))
pred_uhμ = to_element(OutputSpace(uhμ),pred_uμ)
visualise(dir,uhμ,pred_uhμ;variable_name="uh_gridap")

# test 2: GNOs

mesh_size = 1/8
gspace = GraphInducedSpace(test;radius=2.4*mesh_size)
gmodel = KernelNeuralModel(gspace;hidden_layers=(16,16),activation=tanh)
solver = NeuralSolver(fesolver,gmodel;epochs=200,loss=loss,print_every=50)
op = neural_operator(solver,input_space(μ_train),output_space)
pred_uμ = solve(solver,op,input_space(μ))
pred_uhμ = to_element(OutputSpace(uhμ),pred_uμ)
visualise(dir,uhμ,pred_uhμ;variable_name="uh_gno")
