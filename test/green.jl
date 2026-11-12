using DrWatson
using Gridap
using GridapROMs
using GridapROMs.RBSteady
using NonlinearROMs
using Plots
using LinearAlgebra

# Kovachki et al. (2021): for a linear PDE, a single linear kernel layer
# u(x) = ∫ κ(x,y) f(y) dy can represent the solution operator exactly, with
# κ = G the Green function. We test this for -u'' = f on (0,1), u(0) = u(1) = 0,
# whose Green function is known in closed form

green(x,y) = (x + y - abs(x - y)) / 2 - x*y

# forcings must span many modes: with a single (e.g. constant) forcing shape the
# data only fixes ∫ κ(x,y) dy, and G is not identifiable

nmodes = 8
pdomain = ntuple(i -> isodd(i) ? -1 : 1,2*nmodes)
pspace = ParamSpace(pdomain)

domain = (0,1)
h = 1/100
partition = (round(Int,1/h),)
model = CartesianDiscreteModel(domain,partition)

order = 1
degree = 2*order

Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

f(μ) = x -> sum(μ[k]*sin(k*π*x[1]) for k in 1:nmodes)
fμ(μ) = parameterise(f,μ)

g(μ) = x -> 0.0
gμ(μ) = parameterise(g,μ)

stiffness(μ,u,v) = ∫(∇(v)⋅∇(u))dΩ
rhs(μ,v) = ∫(fμ(μ)*v)dΩ

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,2])
trial = ParamTrialFESpace(test,gμ)

feop = LinearParamFEOperator(rhs,stiffness,pspace,trial,test)
fesolver = LinearFESolver(LUSolver())

# FE quantities (snapshots)

μ_train = realisation(pspace;nparams=50)
uhμ_train = solve(fesolver,feop,μ_train)

μ = realisation(pspace;nparams=1)
uhμ = solve(fesolver,feop,μ)

# GNO setup:
# - a complete graph (radius ≥ diam(Ω)), since G is non-local
# - identity activations between/around the kernel layers, so that the whole
#   network is affine in f (κ never sees f at all, since pass_features=false,
#   so giving κ_θ its own nonlinearity below doesn't break this)
# - κ = κ(x,y) only (no features), with an MLP (kernel_activation) since G has
#   a kink at x = y

x = get_coords(test)
features(μ) = FeatureBundle((fμ(μ),))
input_space(μ) = InputSpace(x,features(μ))
output_space = OutputSpace(uhμ_train)

gspace = GraphInducedSpace(test;radius=1.0)
style = IntegralStyle(InducedMeasure(gspace;message_rule=PlainMP()))
gmodel = KernelNeuralModel(
  style;
  hidden_layers=(64,64),
  activation=identity,
  kernel_activation=tanh,
  formatter=DataFormatter(;pass_features=Val(false))
)
solver = NeuralSolver(gmodel;epochs=1000,print_every=100)
op = neural_operator(solver,input_space(μ_train),output_space)

pred_uμ = solve(solver,op,input_space(μ))
uμ = OutputSpace(uhμ).values
println("relative test error: $(norm(pred_uμ - uμ) / norm(uμ))")

# the learned operator's discrete Green function: since the network is affine in
# f, column j is its response to the nodal forcing e_j, minus the response to 0,
# divided by the nodal quadrature weight h

N = length(x)
xs = map(xi -> xi.data[1],x)
coords = repeat(reshape(Float32.(xs),1,N,1),1,1,N+1)
forcings = zeros(Float32,1,N,N+1)
for j in 1:N
  forcings[1,j,j+1] = one(Float32)
end
inputs = (coords=coords,features=forcings)
NonlinearROMs.normalise!(inputs,op.stats)
fixed_data = NonlinearROMs.merge_data(solver,inputs)

lift_layer = op.chain.layers.layer_1
(_,v), = lift_layer(fixed_data,op.parameters.layer_1,op.states.layer_1)
kernel_layer = op.chain.layers.layer_2.kernel
(_,kout), = kernel_layer((fixed_data,v),op.parameters.layer_2.kernel,op.states.layer_2.kernel)
project_layer = op.chain.layers.layer_3.layer
responses_2d, = project_layer(reshape(kout,size(kout,1),:),op.parameters.layer_3,op.states.layer_3)
responses = reshape(responses_2d,size(responses_2d,1),size(kout)[2:end]...)
NonlinearROMs.denormalise!(responses,op.stats)

approx_values = (responses[1,:,2:end] .- responses[1,:,1]) ./ h
values = [green(xi,xj) for xi in xs, xj in xs]
println("relative Green function error: $(norm(approx_values - values) / norm(values))")

# the raw kernel κ(x,y), only equal to G up to the scalings of the lift/projection
# layers and of the data normalisation: compare its shape, not its values

kernel_layer = op.chain.layers[2].kernel
kernel_ps = op.parameters.layer_2.kernel
kernel_st = op.states.layer_2.kernel
x̃s = vec(inputs.coords[1,:,1])

# PlainMP's combine_message is vcat(src,dst,w): the graph is complete (radius=1.0
# spans the whole domain), so every (i,j) pair is a real edge, with the same raw
# (unnormalized) weight |xi-xj| and the same global normalization (the domain's
# diameter among the free-dof points) that GraphInducedSpace's constructor applies
wmax = maximum(xs) - minimum(xs)
pairs = zeros(Float32,3,N*N,1)
for (k,(i,j)) in enumerate(Iterators.product(1:N,1:N))
  pairs[1,k,1] = x̃s[j] # source (y)
  pairs[2,k,1] = x̃s[i] # destination (x)
  pairs[3,k,1] = abs(xs[i] - xs[j]) / wmax # edge weight
end
κ, = kernel_layer.kernel(pairs,kernel_ps,kernel_st)
kernel_values = reshape(κ[1,:,1],N,N)

dir = datadir("green")
mkpath(dir)

hmap(vals,title) = heatmap(xs,xs,vals;
  aspect_ratio=:equal,xlims=(0,1),ylims=(0,1),color=:viridis,
  title,xlabel="y",ylabel="x"
)

savefig(hmap(values,"Green function"),joinpath(dir,"green.png"))
savefig(hmap(approx_values,"GNO Green function"),joinpath(dir,"green_gno.png"))
savefig(hmap(abs.(approx_values - values),"Error"),joinpath(dir,"green_error.png"))
savefig(hmap(kernel_values,"Raw kernel κ(x,y)"),joinpath(dir,"kernel.png"))
