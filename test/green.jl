using DrWatson
using Gridap 
using GridapROMs
using GridapROMs.RBSteady
using NonlinearROMs

pdomain = (1,10)
pspace = ParamSpace(pdomain)

domain = (0,1)
partition = (100,)
model = CartesianDiscreteModel(domain,partition)

order = 1
degree = 2*order

Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

f(μ) = x -> μ[1]
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

μ_train = realisation(pspace;nparams=10)
uhμ_train = solve(fesolver,feop,μ_train)

μ = realisation(pspace;nparams=1)
uhμ = solve(fesolver,feop,μ)

# NO setup

x = get_coords(test)
features(μ) = FeatureBundle((fμ(μ),))
input_space(μ) = InputSpace(x,features(μ))
output_space = OutputSpace(uhμ_train)

# data,stats = NonlinearROMs.fetch_data_and_stats(solver,input_space(μ_train),output_space)
# IS = input_space(μ_train)
# coords,feats = IS.coords,IS.features
# out = output_space.values

# input = NonlinearROMs.fetch_data(solver,input_space(μ_train))
# output = NonlinearROMs.fetch_data(solver,output_space)
# _in = (copy(input.coords),copy(input.features))
# _out = copy(output)
# stats = NonlinearROMs.Normalisation((input,output);normalise=false)
# @assert input.coords == _in[1] && input.features == _in[2] && output == _out
# sc,sf,so = (stats.input_score.scores...,stats.output_score)
# AA = _in[1][1,:,:]
# NonlinearROMs.Normalisation((input,output);normalise=true)
# @assert input.coords[1,:,:] ≈ (AA .- 0.1) ./ (0.9 - 0.1)
# @assert output[1,:,:] ≈ (_out[1,:,:] .- mean(_out,dims=(2,3))) ./ std(_out,dims=(2,3))
# data = (input=NonlinearROMs.merge_data(solver,input),output=output)

# train_state = NonlinearROMs.get_train_state(solver,data)
# dataloader = NonlinearROMs.get_data_loader(solver,data)
# lr_scheduler = NonlinearROMs.get_scheduler(solver)
# loss = NonlinearROMs.get_loss(solver)
# logger = NonlinearROMs.get_logger(solver)
# to_device_batch((x_batch,y_batch)) = (x_batch,y_batch)

# batch_dev = first(dataloader)
# # Lux.Training.single_train_step!(
# #   Lux.AutoEnzyme(),
# #   loss,
# #   batch_dev,
# #   train_state;
# #   return_gradients=Val(false)
# # )

# CIAO
# # pred_uμ = solve(solver,op,input_space(μ))
# # pred_uhμ = to_element(OutputSpace(uhμ),pred_uμ)
# # visualise(dir,uhμ,pred_uhμ;variable_name="uh_gridap")

# test 2: GNOs

mesh_size = 1/10
gspace = GraphInducedSpace(test;radius=2*mesh_size)
gmodel = KernelNeuralModel(gspace;hidden_layers=(1,1),activation=identity)
formatter = DataFormatter(;pass_coords=Val(false))
solver = NeuralSolver(gmodel;formatter,epochs=200,print_every=50)
op = neural_operator(solver,input_space(μ_train),output_space)

green(x,y) = (x + y - abs(x - y)) / 2 - x*y

using Plots

points = Float32.(0.01:0.01:0.99)
values = [green(x,y) for x in points, y in points]

approx_values = zeros(Float32,99,99)
deltai = zeros(Float32,1,99,1)
for i in 1:99
  fill!(deltai,zero(Float32))
  deltai[i] = one(Float32)
  x = vcat(deltai,deltai)
  approx_values[i,:] = l.kernel(x,ps,st)[1]
end

using Plots

h = heatmap(points,points,values,
  aspect_ratio = :equal,
  xlims = (0,1),
  ylims = (0,1),
  color = :viridis,
  title = "Green function",
  xlabel = "x",
  ylabel = "y"
)

heatmap(points,points,approx_values,
  aspect_ratio = :equal,
  xlims = (0,1),
  ylims = (0,1),
  color = :viridis,
  title = "GNO values",
  xlabel = "x",
  ylabel = "y"
)