using DrWatson
using LinearAlgebra
using Lux
using Gridap
using GridapROMs
using GridapROMs.RBSteady
using NonlinearROMs

pdomain = (1,10,1,10,1,10)
pspace = ParamSpace(pdomain)

domain = (0,1,0,1)
partition = (50,50)
model = CartesianDiscreteModel(domain,partition)

order = 2
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

nparams_train = 100
μ_train = realisation(pspace;nparams=nparams_train,sampling=:halton)
uhμ_train = solve(fesolver,feop,μ_train)

μ = realisation(pspace;nparams=5,sampling=:halton,start=nparams_train + 1)
uhμ = solve(fesolver,feop,μ)

# NO setup

dir = datadir("poisson")

x = get_coords(test)
features(μ) = FeatureBundle((aμ(μ),fμ(μ),gμ(μ),hμ(μ)))
input_space(μ) = InputSpace(x,features(μ))
in_train = input_space(μ_train)
in_test = input_space(μ)
output_space = OutputSpace(uhμ_train)
out_test = OutputSpace(uhμ)
uμ = out_test.values

# test 1: Gridap-based spaces

fspace = FunctionSpace(test)
f_epochs = 50
fmodel = KernelNeuralModel(fspace;hidden_layers=(16,16),activation=tanh)
fsolver = NeuralSolver(
  fmodel;
  epochs=f_epochs,
  batch_size=5,
  lr_scheduler=CosineAnnealing(f_epochs;lr_max=Float32(5e-3),lr_min=Float32(1e-5)),
  loss=EnergyLoss(H1(),fspace),
  print_every=10
)
fop = neural_operator(fsolver,in_train,output_space)
pred_uμ = solve(fsolver,fop,in_test)
println("FunctionSpace relative L2 test error: $(norm(pred_uμ - uμ) / norm(uμ))")
pred_uhμ = to_element(out_test,pred_uμ)
visualise(dir,uhμ,pred_uhμ;variable_name="uh_gridap")

# test 2: GNOs (subsampled training via Sampler + zero-shot full-resolution inference)

# Quasi-uniform subgrid: 1 out of every 4 DoFs per axis (h_sub = 0.04, N_sub = 650)
dof_res = 50 * order
stride = 4
sub_ids = findall(p -> (round(Int,p[1] * dof_res) % stride == 0) && (round(Int,p[2] * dof_res) % stride == 0),x)
r_phys = 2 * stride / dof_res

gspace_sub = GraphInducedSpace(x[sub_ids];radius=r_phys)
g_epochs = 600
gmodel = KernelNeuralModel(
  gspace_sub;
  hidden_layers=(24,24,24),
  activation=tanh,
  formatter=FeatureKernelFormatter(;sampler=Sampler(sub_ids))
)
gsolver = NeuralSolver(
  gmodel;
  epochs=g_epochs,
  batch_size=10,
  lr_scheduler=CosineAnnealing(g_epochs;lr_max=Float32(5e-3),lr_min=Float32(1e-5)),
  loss=Lux.MSELoss(),
  print_every=100
)
gop_sub = neural_operator(gsolver,in_train,output_space)

# Transfer trained operator to the full 50x50 Q2 mesh with the same physical radius
gspace_full = GraphInducedSpace(test;radius=r_phys)
gop_full = replace_space(gop_sub,gspace_full)
gsolver_infer = NeuralSolver(gmodel;formatter=DataFormatter())
pred_gno = similar(uμ)
for k in axes(uμ,3)
  in_k = InputSpace(in_test.coords[:,:,k:k],in_test.features[:,:,k:k])
  pred_gno[:,:,k:k] = solve(gsolver_infer,gop_full,in_k)
end
println("GraphInducedSpace relative L2 test error: $(norm(pred_gno - uμ) / norm(uμ))")
pred_uhμ = to_element(out_test,pred_gno)
visualise(dir,uhμ,pred_uhμ;variable_name="uh_gno")