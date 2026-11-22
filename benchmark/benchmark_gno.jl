using BenchmarkTools
using Random
using LinearAlgebra
using Graphs
using Lux
using Gridap
using Gridap.FESpaces
using GridapROMs
using GridapROMs.RBSteady
using GridapSolvers
using GridapSolvers.LinearSolvers
using NonlinearROMs

using NonlinearROMs: TData,ZScore,_integration,_kv,build_model

println("==========================================================")
println("   NonlinearROMs.jl - Kernel Operator Benchmarks")
println("==========================================================")

# Reproducibility
Random.seed!(42)

# ==========================================================
# 1. Graph and data setup (50x50 Cartesian mesh)
# ==========================================================

domain = (0,1,0,1)
partition = (50,50)

model = CartesianDiscreteModel(domain,partition)
reffe = ReferenceFE(lagrangian,Float64,1)

V = TestFESpace(
  model,
  reffe;
  conformity=:H1,
  dirichlet_tags=[1,3,7]
)

mesh_size = 1 / 50
gspace = GraphInducedSpace(V;radius=2.0 * mesh_size)
fspace = FunctionSpace(V)
meas = InducedMeasure(gspace;message_rule=PlainMP())
g = get_graph(gspace)

N = length(get_coords(gspace))
E = ne(g)

K = 10       # Number of training samples
D_in = 6     # 2 spatial coordinates + 4 PDE features
D_hid = 16   # Hidden state channels

println("Mesh configuration: N = $N nodes | E = $E edges | K = $K samples\n")

fixed_data = rand(TData,D_in,N,K)
dummy_out = rand(TData,1,N,K)
pred_dummy = rand(TData,1,N,K)

edge_msgs = rand(TData,D_hid,E,K)
k_full = rand(TData,D_hid * D_hid,E,K)

# ==========================================================
# 2. CPU micro-benchmarks (aligned on PlainMP)
# ==========================================================

println("1. ZScore computation and in-place normalisation (D=$D_in, N=$N, K=$K):")
b_zscore = @benchmark ZScore(d;normalise=true) setup=(d = copy($fixed_data))
display(b_zscore)

println("\n2. Graph message gathering (gather_data, PlainMP):")
b_gather = @benchmark gather_data($fixed_data,$meas)
display(b_gather)

println("\n3. Graph message integration (_integration = scatter_data + weighting):")
b_integ = @benchmark _integration($edge_msgs,$meas)
display(b_integ)

println("\n4a. Kernel-state product _kv (Full-Matrix $D_hid x $D_hid per edge):")
b_kv_full = @benchmark _kv($k_full,$edge_msgs)
display(b_kv_full)

println("\n4b. Kernel-state product _kv (Channel-Wise $D_hid per edge):")
b_kv_cw = @benchmark _kv($edge_msgs,$edge_msgs)
display(b_kv_cw)

println("\n4c. EnergyLoss(H1(), fspace) evaluation (N=$N, K=$K):")
h1_loss_bench = EnergyLoss(H1(),fspace)
b_eloss = @benchmark $h1_loss_bench($pred_dummy,$dummy_out)
display(b_eloss)

# ==========================================================
# 3. CPU full-chain forward pass (aligned on PlainMP)
# ==========================================================

gno_model = KernelNeuralModel(
  IntegralStyle(meas);
  hidden_layers=(16,16,16),
  activation=tanh
)

chain = build_model(gno_model,(fixed_data,dummy_out))
ps,st = Lux.setup(Random.default_rng(),chain)

println("\n5a. Full GNO Chain Forward Pass (2 KernelNeuralLayers, PlainMP, FeatureKernel, CPU):")
b_chain = @benchmark $chain($fixed_data,$ps,$st)
display(b_chain)

gno_spatial = KernelNeuralModel(
  IntegralStyle(meas);
  hidden_layers=(16,16,16),
  activation=tanh,
  formatter=SpatialKernelFormatter()
)
chain_sp = build_model(gno_spatial,(fixed_data,dummy_out))
ps_sp,st_sp = Lux.setup(Random.default_rng(),chain_sp)

println("\n5b. Full GNO Chain Forward Pass (2 KernelNeuralLayers, PlainMP, SpatialKernel, CPU):")
b_chain_sp = @benchmark $chain_sp($fixed_data,$ps_sp,$st_sp)
display(b_chain_sp)

# ==========================================================
# 4. Reactant / XLA training benchmark (PlainMP)
# ==========================================================

println("\n==========================================================")
println("6. GNO Training on Reactant / XLA (PlainMP, 2 layers)")
println("==========================================================")

XLA_EPOCHS = 200

xla_solver = NeuralSolver(
  gno_model;
  epochs=XLA_EPOCHS,
  batch_size=0,
  verbose=true,
  print_every=50
)

xla_logger = NonlinearROMs.get_logger(xla_solver)

function run_xla_training(solver,inputs,targets)
  t0 = time()
  trained_op = NonlinearROMs.train(solver,(inputs,targets),nothing)
  return trained_op,time() - t0
end

println("\nStarting XLA training ($XLA_EPOCHS epochs)...")
xla_op,xla_call_time = run_xla_training(xla_solver,fixed_data,dummy_out)
xla_end_time = time()

xla_first_epoch_time = xla_logger.t_start_fast - xla_logger.t_start
xla_after_first_epoch_time = xla_end_time - xla_logger.t_start_fast
xla_training_loop_time = xla_end_time - xla_logger.t_start
xla_setup_time = max(0.0,xla_call_time - xla_training_loop_time)

println("\nXLA Timing Summary")
println("------------------------------")
println("Epochs:                         $XLA_EPOCHS")
println("Training samples:               $K")
println("First epoch (JIT + training):   $(round(xla_first_epoch_time,digits=3)) s")
println("Training after first epoch:     $(round(xla_after_first_epoch_time,digits=3)) s")
println("Training loop total:            $(round(xla_training_loop_time,digits=3)) s")
println("Setup and other overhead:       $(round(xla_setup_time,digits=3)) s")
println("Total call time:                $(round(xla_call_time,digits=3)) s")

# ==========================================================
# 5. Post-training sanity check
# ==========================================================

println("\n7. Post-training sanity check:")
pred,_ = xla_op.chain(fixed_data,xla_op.parameters,xla_op.states)
@assert size(pred) == size(dummy_out)
@assert all(isfinite,pred)
println("SUCCESS! Output shape: $(size(pred))")

# ==========================================================
# 6. GNO Accuracy & Optimization Diagnostic (2D Poisson 25x25)
# ==========================================================

println("\n==========================================================")
println("8. GNO Accuracy & Optimization Diagnostic (2D Poisson 25x25)")
println("==========================================================")

pdomain_d = (1,10,1,10,1,10)
pspace_d = ParamSpace(pdomain_d)
model_d = CartesianDiscreteModel((0,1,0,1),(25,25))
Ω_d = Triangulation(model_d)
dΩ_d = Measure(Ω_d,2)
Γn_d = BoundaryTriangulation(model_d,tags=[8])
dΓn_d = Measure(Γn_d,2)

a_d(μ) = x -> exp(-x[1]/sum(μ))
f_d(μ) = x -> 1.0
g_d(μ) = x -> μ[1]*exp(-x[1]/μ[2])
h_d(μ) = x -> abs(cos(μ[3]*x[2]))

stiff_d(μ,u,v) = ∫(parameterise(a_d,μ)*∇(v)⋅∇(u))dΩ_d
rhs_d(μ,v) = ∫(parameterise(f_d,μ)*v)dΩ_d + ∫(parameterise(h_d,μ)*v)dΓn_d

V_d = TestFESpace(Ω_d,ReferenceFE(lagrangian,Float64,1);conformity=:H1,dirichlet_tags=[1,3,7])
U_d = ParamTrialFESpace(V_d,μ -> parameterise(g_d,μ))
feop_d = LinearParamFEOperator(rhs_d,stiff_d,pspace_d,U_d,V_d)
fesolver_d = LinearFESolver(LUSolver())

μ_tr = realisation(pspace_d;nparams=30)
uh_tr = solve(fesolver_d,feop_d,μ_tr)
μ_te = realisation(pspace_d;nparams=10,start=31)
uh_te = solve(fesolver_d,feop_d,μ_te)

x_d = get_coords(V_d)
feats_d(μ) = FeatureBundle((parameterise(a_d,μ),parameterise(f_d,μ),parameterise(g_d,μ),parameterise(h_d,μ)))
in_tr = InputSpace(x_d,feats_d(μ_tr))
out_tr = OutputSpace(uh_tr)
u_tr_true = out_tr.values
in_te = InputSpace(x_d,feats_d(μ_te))
u_te_true = OutputSpace(uh_te).values

h_d_size = 1 / 25
bnd_mask = [min(p[1],1.0 - p[1],p[2],1.0 - p[2]) <= 1.5 * h_d_size for p in x_d]
int_mask = .!bnd_mask

fspace_d = FunctionSpace(V_d)
h1_loss_d = EnergyLoss(H1(),fspace_d)

# Tuple fields: (label, radius, rule, hidden_layers, epochs, lr_max, loss_fn, weight_decay)
configs_d = [
  ("1-hop r=2h DiffMP 300ep",      2*h_d_size,DifferenceMP(),(16,16),   300, TData(1e-3),Lux.MSELoss(),0.0),
  ("1-hop r=2h PlainMP 300ep",     2*h_d_size,PlainMP(),     (16,16),   300, TData(1e-3),Lux.MSELoss(),0.0),
  ("1-hop r=2h PlainMP 1500ep",    2*h_d_size,PlainMP(),     (16,16),   1500,TData(5e-3),Lux.MSELoss(),0.0),
  ("2-hop r=3h PlainMP 1500ep",    3*h_d_size,PlainMP(),     (24,24,24),1500,TData(5e-3),Lux.MSELoss(),0.0),
  ("1-hop r=2h PlainMP H1-loss",   2*h_d_size,PlainMP(),     (16,16),   1500,TData(5e-3),h1_loss_d,    0.0),
  ("1-hop r=2h AnchoredMP H1+wd",  2*h_d_size,AnchoredMP(),  (16,16),   1500,TData(5e-3),h1_loss_d,    1e-4),
  ("2-hop r=3h AnchoredMP wd",     3*h_d_size,AnchoredMP(),  (24,24,24),1500,TData(5e-3),Lux.MSELoss(),1e-4),
]

rows_d = String[]
for (label,rad,rule,hlayers,ep,lr,lfn,wd) in configs_d
  gs = GraphInducedSpace(V_d;radius=rad)
  meas_cfg = InducedMeasure(gs;message_rule=rule)
  gm = KernelNeuralModel(IntegralStyle(meas_cfg);hidden_layers=hlayers,activation=tanh)
  sched = CosineAnnealing(ep;lr_max=lr)
  slv = NeuralSolver(gm;epochs=ep,lr_scheduler=sched,loss=lfn,weight_decay=wd,verbose=true,print_every=ep + 1)
  lg = NonlinearROMs.get_logger(slv)
  op_cfg = neural_operator(slv,in_tr,out_tr)
  t_end = time()
  t_comp = round(lg.t_start_fast - lg.t_start,digits=2)
  t_trn = round(t_end - lg.t_start_fast,digits=2)
  pred_tr = solve(slv,op_cfg,in_tr)
  pred_te = solve(slv,op_cfg,in_te)
  tr_l2 = round(norm(pred_tr - u_tr_true) / norm(u_tr_true),digits=5)
  te_l2 = round(norm(pred_te - u_te_true) / norm(u_te_true),digits=5)
  in_l2 = round(norm(pred_te[:,int_mask,:] - u_te_true[:,int_mask,:]) / norm(u_te_true[:,int_mask,:]),digits=5)
  bn_l2 = round(norm(pred_te[:,bnd_mask,:] - u_te_true[:,bnd_mask,:]) / norm(u_te_true[:,bnd_mask,:]),digits=5)
  row = "$(rpad(label,30)) | E=$(lpad(ne(get_graph(gs)),6)) | comp=$(lpad(t_comp,6))s | trn=$(lpad(t_trn,6))s | trL2=$(lpad(tr_l2,7)) | teL2=$(lpad(te_l2,7)) | intL2=$(lpad(in_l2,7)) | bndL2=$(lpad(bn_l2,7))"
  push!(rows_d,row)
end

println("\n--- Summary: 2D Poisson (25x25) ---")
for r in rows_d
  println(r)
end

# ==========================================================
# 7. Non-Local Multi-Mode 2D Poisson Diagnostic (15x15)
# ==========================================================

println("\n==========================================================")
println("9. Non-Local Multi-Mode 2D Poisson Diagnostic (15x15)")
println("==========================================================")

pdomain_nl = (-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0)
pspace_nl = ParamSpace(pdomain_nl)
model_nl = CartesianDiscreteModel((0,1,0,1),(15,15))
Ω_nl = Triangulation(model_nl)
dΩ_nl = Measure(Ω_nl,2)

f_nl(μ) = x -> (
  μ[1]*sin(π*x[1])*sin(π*x[2]) +
  μ[2]*sin(2*π*x[1])*sin(π*x[2]) +
  μ[3]*sin(π*x[1])*sin(2*π*x[2]) +
  μ[4]*sin(2*π*x[1])*sin(2*π*x[2])
)
g_nl(μ) = x -> 0.0

stiff_nl(μ,u,v) = ∫(∇(v)⋅∇(u))dΩ_nl
rhs_nl(μ,v) = ∫(parameterise(f_nl,μ)*v)dΩ_nl

V_nl = TestFESpace(Ω_nl,ReferenceFE(lagrangian,Float64,1);conformity=:H1,dirichlet_tags="boundary")
U_nl = ParamTrialFESpace(V_nl,μ -> parameterise(g_nl,μ))
feop_nl = LinearParamFEOperator(rhs_nl,stiff_nl,pspace_nl,U_nl,V_nl)

μ_nl_tr = realisation(pspace_nl;nparams=40,sampling=:halton)
uh_nl_tr = solve(fesolver_d,feop_nl,μ_nl_tr)
μ_nl_te = realisation(pspace_nl;nparams=10,sampling=:halton,start=41)
uh_nl_te = solve(fesolver_d,feop_nl,μ_nl_te)

x_nl = get_coords(V_nl)
feats_nl(μ) = FeatureBundle((parameterise(f_nl,μ),))
in_nl_tr = InputSpace(x_nl,feats_nl(μ_nl_tr))
out_nl_tr = OutputSpace(uh_nl_tr)
u_nl_tr_true = out_nl_tr.values
in_nl_te = InputSpace(x_nl,feats_nl(μ_nl_te))
u_nl_te_true = OutputSpace(uh_nl_te).values

h_nl = 1 / 15
configs_nl = [
  ("NL 1-hop r=2h PlainMP 1000ep",  2*h_nl,PlainMP(),   (16,16),      FeatureKernelFormatter(),1000,TData(5e-3)),
  ("NL 3-hop r=3h PlainMP 1000ep",  3*h_nl,PlainMP(),   (24,24,24,24),FeatureKernelFormatter(),1000,TData(5e-3)),
  ("NL 3-hop r=3h AnchoredMP Spat", 3*h_nl,AnchoredMP(),(24,24,24,24),SpatialKernelFormatter(),1000,TData(5e-3)),
]

rows_nl = String[]
for (label,rad,rule,hlayers,fmt,ep,lr) in configs_nl
  gs = GraphInducedSpace(V_nl;radius=rad)
  meas_cfg = InducedMeasure(gs;message_rule=rule)
  gm = KernelNeuralModel(IntegralStyle(meas_cfg);hidden_layers=hlayers,activation=tanh,formatter=fmt)
  sched = CosineAnnealing(ep;lr_max=lr)
  slv = NeuralSolver(gm;epochs=ep,lr_scheduler=sched,verbose=true,print_every=ep + 1)
  lg = NonlinearROMs.get_logger(slv)
  op_cfg = neural_operator(slv,in_nl_tr,out_nl_tr)
  t_end = time()
  t_comp = round(lg.t_start_fast - lg.t_start,digits=2)
  t_trn = round(t_end - lg.t_start_fast,digits=2)
  pred_tr = solve(slv,op_cfg,in_nl_tr)
  pred_te = solve(slv,op_cfg,in_nl_te)
  tr_l2 = round(norm(pred_tr - u_nl_tr_true) / norm(u_nl_tr_true),digits=5)
  te_l2 = round(norm(pred_te - u_nl_te_true) / norm(u_nl_te_true),digits=5)
  row = "$(rpad(label,30)) | E=$(lpad(ne(get_graph(gs)),6)) | comp=$(lpad(t_comp,6))s | trn=$(lpad(t_trn,6))s | trL2=$(lpad(tr_l2,7)) | teL2=$(lpad(te_l2,7))"
  push!(rows_nl,row)
end

println("\n--- Summary: Non-Local Multi-Mode 2D Poisson (15x15) ---")
for r in rows_nl
  println(r)
end

println("\n==========================================================")
println("Benchmark completed.")
println("==========================================================")