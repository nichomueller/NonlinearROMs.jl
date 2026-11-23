# # Tutorial: an autoregressive Neural Operator for the parametric heat equation
#
# This tutorial extends the
# [Poisson tutorial](@ref "Tutorial: a Neural Operator surrogate for a parametric Poisson problem")
# to a **transient** problem, and assumes you have read it: parameter spaces,
# `ParamFunction`s, `InputSpace`/`OutputSpace`, `KernelNeuralModel` and
# `EnergyLoss` are used exactly as there. What is new here is time. Instead of
# learning ``\mu\mapsto u_h(\mu)`` at once, the network learns how to advance
# the solution in time, and is then rolled out autoregressively from the initial
# condition. The training strategy follows Brandstetter et al., *Message Passing
# Neural PDE Solvers* (2022): temporal bundling plus the pushforward trick.
#
# The problem is the heat equation on the unit square,
# ```math
# \partial_t u - \nabla\cdot(a(\mu,t)\nabla u) = f(\mu,t) \ \text{in } \Omega\times(0,T],
# \qquad u = g(\mu,t) \ \text{on } \Gamma_D, \qquad
# a(\mu,t)\nabla u\cdot n = h(\mu,t) \ \text{on } \Gamma_N,
# ```
# with ``u(0) = 0``, discretised in time by the ``\theta``-method.

using DrWatson
using Gridap
using GridapROMs
using GridapROMs.ParamDataStructures
using GridapROMs.RBSteady
using NonlinearROMs

# ## 1. The transient parametric problem
#
# A [`TransientParamSpace`](@ref) couples the parameter box with the time
# grid; coefficients now depend on `(μ,t)` and are lifted with
# `parameterise(f,μ,t)`. The rest is the standard `GridapROMs` setup for a
# linear transient problem, with a stiffness and a mass form.

θ = 0.5
dt = 0.01
t0 = 0.0
tf = 10*dt
tdomain = t0:dt:tf
pdomain = (1,10,1,10,1,10)
ptspace = TransientParamSpace(pdomain,tdomain)

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

fesolver = ThetaMethod(LUSolver(),dt,θ)
feop = TransientLinearParamFEOperator(rhs,(stiffness,mass),ptspace,trial,test)

# ## 2. Snapshots
#
# `solve` returns a `TransientParamFESolution`, an iterator over the time
# steps ``t_1,\ldots,t_{N_t}``, each holding the solutions of every parameter.
# One set of parameters is used for training and another for testing.

dir = datadir("heat_equation")
mkpath(dir)

μt_train = realisation(ptspace;nparams=50)
μ_train = get_params(μt_train)
uhμ_train = solve(fesolver,feop,μt_train,uh0μ)

μt = realisation(ptspace;nparams=50)
μ = get_params(μt)
uhμ = solve(fesolver,feop,μt,uh0μ)

# ## 3. Temporal bundling: `RolloutData`
#
# [`RolloutData`](@ref)`(data,times,step)` cuts the time steps into
# ``n_c`` disjoint **clusters** of `step` consecutive times. Each cluster is
# split into two **stages** of `step÷2` time slots each. Writing ``F^c_k`` for
# the features and ``s^c_k`` for the states at stage ``k`` of cluster ``c``, the
# network ``N`` learns to map the previous state block, together with the
# features at the times being predicted, to the next state block:
# ```math
# N([x;F^c_2;s^c_1]) \approx s^c_2, \qquad N([x;F^c_1;s^{c-1}_2]) \approx s^c_1.
# ```
# With `step=2` every stage holds a single time step, so ``N`` is a one-step
# time integrator. A larger `step` makes ``N`` predict several time steps at
# once, i.e. temporal bundling.
#
# Evaluated in an [`InputSpace`](@ref) or [`OutputSpace`](@ref), a
# `RolloutData` becomes [`TwoStageBuckets`](@ref): one `(D,N,K)` array per
# stage, with the time slots of a stage stacked along the channels and the
# samples ordered cluster by cluster (``K = n_c\,n_p``).

x = get_coords(test)
times = collect(dt:dt:tf)

features(μ) = RolloutData(
  FeatureBundle((aμt(μ,times),fμt(μ,times),gμt(μ,times),hμt(μ,times))),
  times,2
)
input_space(μ) = InputSpace(x,features(μ))
output_space = OutputSpace(RolloutData(uhμ_train,times,2))

# ## 4. Training: one-step and pushforward losses
#
# A [`RolloutNeuralSolver`](@ref) takes the same arguments as a
# [`NeuralSolver`](@ref), and trains on
# ```math
# \sum_c \ell\big(N([x;F^c_2;s^c_1]),s^c_2\big)
# + \lambda\sum_{c\ge2}\ell\big(N([x;F^c_1;\mathrm{sg}(N([x;F^{c-1}_2;s^{c-1}_1]))]),s^c_1\big).
# ```
# The first term is the usual one-step (teacher-forced) loss on ground-truth
# states. In the second, the **pushforward** term, the network's input is one
# of its own predictions. A network trained on ground truth alone never sees
# its own errors and tends to drift once rolled out; the pushforward term
# exposes it to the input distribution it meets at inference. The
# stop-gradient ``\mathrm{sg}`` means that gradients only flow through the
# outer application, which keeps training as cheap as one-step training. Every
# batch mixes clusters, and both stages share a single normalisation, so a
# prediction can be fed back as an input on the same scale.
#
# `warmup_epochs` trains the one-step term alone first, so the pushforward
# term starts from sensible predictions; `λ` weights it.
#
# The network sees the previous state as extra input channels: its input
# dimension is that of the features plus that of the state.

fspace = FunctionSpace(test)
fmodel = KernelNeuralModel(fspace;hidden_layers=(16,16),activation=tanh)
solver = RolloutNeuralSolver(fmodel;epochs=200,loss=EnergyLoss(H1(),fspace),print_every=50)
op = neural_operator(solver,input_space(μ_train),output_space)

# ## 5. Rollout from the initial condition
#
# At inference no true state is available besides the initial condition, so
# every prediction is fed back:
# ```math
# \hat y^0_2 = u_0, \qquad \hat y^c_1 = N([x;F^c_1;\hat y^{c-1}_2]), \qquad
# \hat y^c_2 = N([x;F^c_2;\hat y^c_1]).
# ```
# `solve(solver,op,input_space,initial_output_space)` does exactly this (see
# [`rollout`](@ref)), with the initial state evaluated in an `OutputSpace`.
# The prediction is a `TwoStageBuckets`, which [`to_element`](@ref) turns back
# into a `ParamFEFunction` over every (parameter, time) pair.
#
# !!! note
#     The rollout times start at ``t_1``, so the transition
#     ``u_0\to u(t_1)`` never appears in the training data: the first step is
#     an extrapolation. With `step > 2`, the initial state block needs
#     `step÷2` time slots, of which only ``u_0`` is known.

output_space0 = OutputSpace(uh0μ(μ))
pred_uμ = solve(solver,op,input_space(μ),output_space0)

true_uμ = OutputSpace(RolloutData(uhμ,times,2)).values
rel_err(a,b) = sqrt(sum(abs2,a .- b)/sum(abs2,b))
println("relative rollout error, first stage:  $(rel_err(pred_uμ.first_stage,true_uμ.first_stage))")
println("relative rollout error, second stage: $(rel_err(pred_uμ.second_stage,true_uμ.second_stage))")

# !!! note
#     With the small network and training budget used here, the rollout is
#     only qualitatively right. Even the one-step predictions (from true
#     states) are far from accurate, and their errors compound over the
#     rollout. One time step changes the solution very little, so the network
#     has to learn a map close to the identity. A wider network, more epochs,
#     or a residual formulation that predicts the increment
#     ``\hat y = s + N(\ldots)``, as Brandstetter et al. do, help considerably.

# ## 6. Visualisation
#
# For a transient solution, [`visualise`](@ref) writes one `.vtu` file per
# time step, plus a `.pvd` collection that ParaView opens as a time series.
# The fields are the true solution, the prediction and the error, for the
# first test parameter by default (`param_id`).

pred_uhμ = to_element(OutputSpace(uhμ),pred_uμ)
visualise(dir,uhμ,pred_uhμ;variable_name="uh_gridap")
