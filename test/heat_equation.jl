using Gridap 
using GridapROMs
using GridapROMs.RBSteady
using NonlinearROMs

pdomain = (1,10,1,10,1,10)
pspace = ParamSpace(pdomain)

domain = (0,1,0,1)
partition = (5,5)
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

θ = 0.5
dt = 0.01
t0 = 0.0
tf = 10*dt
tdomain = t0:dt:tf

ptspace = TransientParamSpace(pdomain,tdomain)

fesolver = ThetaMethod(LUSolver(),dt,θ)
feop = TransientLinearParamFEOperator(rhs,(stiffness,mass),ptspace,trial,test)

μ = realisation(pspace;nparams=50)
μt = TransientRealisation(μ,collect(tdomain))
uhμ = solve(fesolver,feop,μt,uh0μ)

using GridapROMs.ParamDataStructures
using Gridap.Algebra

fspace = FunctionSpace(test)
gspace = GraphInducedSpace(fspace;radius=0.3)

times = collect(tdomain)
states = [uh0μ(μ);collect_trajectory(uhμ)]
features_fn(t) = FeatureBundle((aμt(μ,t),fμt(μ,t),gμt(μ,t),hμt(μ,t)))
input_space,output_space = rollout_data(fspace,features_fn,times,states)

model = KernelNeuralModel(gspace;hidden_layers=(16,16),activation=tanh)
solver = NeuralSolver(fesolver,model;epochs=200,print_every=50)

# offline phase: train the one-step (autoregressive) operator
op = neural_operator(solver,input_space,output_space)

# online phase: roll the trained operator out over the whole trajectory, starting
# from the true initial condition -- each step only ever sees its own prediction
predicted_states = rollout(solver,op,fspace,features_fn,times,states[1])