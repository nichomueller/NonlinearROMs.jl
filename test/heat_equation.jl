using Gridap 
using GridapROMs
using GridapROMs.ParamDataStructures
using GridapROMs.RBSteady
using NonlinearROMs
using DrWatson

θ = 0.5
dt = 0.01
t0 = 0.0
tf = 10*dt
tdomain = t0:dt:tf
pdomain = (1,10,1,10,1,10)
ptspace = TransientParamSpace(pdomain,tdomain)

L,W = 1,1
N = 5
domain = (0,L,0,W)
partition = (N,N)
model = CartesianDiscreteModel(domain,partition)

order = 1
degree = 2*order

Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=[8])
dΓn = Measure(Γn,degree)

a(μ,t) = x -> 1+exp(-sin(t)^2*x[1]/sum(μ))
aμt(μ,t) = parameterise(a,μ,t)

f(μ,t) = x -> 1.
fμt(μ,t) = parameterise(f,μ,t)

h(μ,t) = x -> abs(cos(t/μ[3]))
hμt(μ,t) = parameterise(h,μ,t)

g(μ,t) = x -> μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
gμt(μ,t) = parameterise(g,μ,t)

u0(μ) = x -> 0.0
u0μ(μ) = parameterise(u0,μ)

stiffness(μ,t,u,v) = ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ
mass(μ,t,uₜ,v) = ∫(v*uₜ)dΩ
rhs(μ,t,v) = ∫(fμt(μ,t)*v)dΩ + ∫(hμt(μ,t)*v)dΓn

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
trial = TransientTrialParamFESpace(test,gμt)

uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

fesolver = ThetaMethod(LUSolver(),dt,θ)
feop = TransientLinearParamFEOperator(rhs,(stiffness,mass),ptspace,trial,test)

dir = datadir("heat_equation")
mkpath(dir)

μt_train = realisation(ptspace;nparams=50)
μ_train = get_params(μt_train)
uhμ_train = solve(fesolver,feop,μt_train,uh0μ)

μt = realisation(ptspace;nparams=50)
μ = get_params(μt)
uhμ = solve(fesolver,feop,μt,uh0μ)

x = get_coords(test)
times = collect(dt:dt:tf)

features(μ) = RolloutData(
  FeatureBundle((aμt(μ,times),fμt(μ,times),gμt(μ,times),hμt(μ,times))),
  times,2
)
input_space(μ) = InputSpace(x,features(μ))
output_space = OutputSpace(RolloutData(uhμ_train,times,2))
output_space0 = OutputSpace(uh0μ(μ))

# test 1: Gridap-based spaces

fspace = FunctionSpace(test)
fmodel = KernelNeuralModel(fspace;hidden_layers=(16,16),activation=tanh)
solver = RolloutNeuralSolver(fmodel;epochs=200,loss=EnergyLoss(H1(),fspace),print_every=50)
op = neural_operator(solver,input_space(μ_train),output_space)
pred_uμ = solve(solver,op,input_space(μ),output_space0)
pred_uhμ = to_element(OutputSpace(uhμ),pred_uμ)
visualise(dir,uhμ,pred_uhμ;variable_name="uh_gridap")

# test 2: GNOs

mesh_size = 1/N
gspace = GraphInducedSpace(test;radius=mesh_size)
gmodel = KernelNeuralModel(gspace;hidden_layers=(16,16),activation=tanh)
solver = RolloutNeuralSolver(gmodel;epochs=200,loss=EnergyLoss(H1(),fspace),print_every=50)
op = neural_operator(solver,input_space(μ_train),output_space)
pred_uμ = solve(solver,op,input_space(μ),output_space0)
pred_uhμ = to_element(OutputSpace(uhμ),pred_uμ)
visualise(dir,uhμ,pred_uhμ;variable_name="uh_gno")

# test 3: FNOs

kmodel = KernelNeuralModel(FourierStyle(truncation=12);hidden_layers=(16,16),activation=tanh)
solver = RolloutNeuralSolver(kmodel;epochs=200,loss=EnergyLoss(H1(),fspace),print_every=50)
op = neural_operator(solver,input_space(μ_train),output_space)
pred_uμ = solve(solver,op,input_space(μ),output_space0)
pred_uhμ = to_element(OutputSpace(uhμ),pred_uμ)
visualise(dir,uhμ,pred_uhμ;variable_name="uh_fno")