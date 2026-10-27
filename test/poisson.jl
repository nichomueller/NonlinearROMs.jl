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

trian_res = (Ω,Γn)
trian_stiffness = (Ω,)
domains = FEDomains(trian_res,trian_stiffness)

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
trial = ParamTrialFESpace(test,gμ)

feop = LinearParamFEOperator(rhs,stiffness,pspace,trial,test,domains)
fesolver = LinearFESolver(LUSolver())

μ = realisation(pspace;nparams=50)
uhμ = solve(fesolver,feop,μ)

using GridapROMs.ParamDataStructures
using Gridap.Algebra

# bundle of parameterised functions evaluated as the kernel's fixed (x,a(x)) input,
# alongside x itself (see InputSpace below)
features = FeatureBundle((aμ(μ),fμ(μ),gμ(μ),hμ(μ)))

# the discretisation: x -> a graph induced by the test space's dof coordinates,
# used both for the input sampling (InputSpace) and for the non-local integral
# kernel itself (IntegralStyle wraps a GraphMeasure over this same graph)
fspace = FunctionSpace(test)
gspace = GraphInducedSpace(fspace;radius=0.3)

input_space = InputSpace(fspace,features)

# the output space: the FE solution itself, sampled at the same dof coordinates
output_space = OutputSpace(uhμ)

# a kernel-integral neural operator (Kovachki et al.): lifting -> one non-local
# KernelNeuralLayer (graph-based GNO integral kernel) -> projection
model = KernelNeuralModel(fspace;hidden_layers=(16,16),activation=tanh)
X = assemble_matrix((u,v) -> ∫(∇(v)⋅∇(u))dΩ,trial,test)
solver = NeuralSolver(fesolver,model;epochs=200,loss=EnergyLoss(X),print_every=50)

# offline phase: train the operator
op = neural_operator(solver,input_space,output_space)

# online phase, reconstructed as an actual FEFunction (dirichlet values come from
# output_space's own fe space, which must be trial(μ), not test -- see OutputSpace)
α = realisation(pspace;nparams=1)
uhα = solve(fesolver,feop,α)
online_features = FeatureBundle((aμ(α),fμ(α),gμ(α),hμ(α)))
online_input_space = InputSpace(fspace,online_features)
pred_uhα = solve(solver,op,online_input_space)

#=
problem statement: 

given 
domains Ω1,...,Ωn
triangulations T1,...,Tn
discrete spaces V1,...,Vn
sets of parameters μ1,...,μn
parameterized functions {a_i(μ),f_i(μ),g_i(μ),h_i(μ)}_{i=1}^n
we want to find the map
\mathcal{N}: (x*,a*(μ),f*(μ),g*(μ),h*(μ)) -> u*(μ)
where x* is a point in the domain Ω*,a*(μ),f*(μ),g*(μ),h*(μ) are the parameterized functions defined on Ω*,
and u*(μ) is the solution of the PDE defined on Ω* with parameters μ* and functions a*(μ),f*(μ),g*(μ),h*(μ).
so i cannot pass feop; input space (x + bundles + param space) + discrete space (integration) -> output space (usually == discr space)
neural solver should store the rest: neural model, data formatter, training-related things, 
build_map(neural_solver,input_space,output_space,integration_space=output_space)
=#