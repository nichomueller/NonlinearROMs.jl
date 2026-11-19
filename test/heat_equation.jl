using Gridap 
using GridapROMs
using GridapROMs.ParamDataStructures
using GridapROMs.RBSteady
using NonlinearROMs

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

# test 1: Gridap-based spaces

fspace = FunctionSpace(test)
fmodel = KernelNeuralModel(fspace;hidden_layers=(16,16),activation=tanh)
solver = NeuralSolver(fmodel;epochs=200,loss=EnergyLoss(H1(),fspace),print_every=50)
op = neural_operator(solver,input_space(μ_train),output_space)
pred_uμ = solve(solver,op,input_space(μ))
pred_uhμ = to_element(OutputSpace(uhμ),pred_uμ)
visualise(dir,uhμ,pred_uhμ;variable_name="uh_gridap")

ispace = input_space(μ_train)
# neural_operator(solver,ispace,output_space)
data,stats = NonlinearROMs.fetch_data_and_stats(
  solver,ispace,output_space
)

function rollout_train_model!(train_state,dataloader,stats,lr_scheduler,loss,to_device_batch;logger)
  init!(logger)

  Reactant.with_config(;dot_general_precision=Reactant.PrecisionConfig.HIGH) do
    for epoch in 1:logger.max_epochs
      local current_loss = zero(TData)

      for raw_batch in dataloader
        batch_dev = to_device_batch(raw_batch)

        _,loss_val,_,train_state = Lux.Training.single_train_step!(
          Lux.AutoEnzyme(),
          loss,
          batch_dev,
          train_state;
          return_gradients=Val(false)
        )
        current_loss += TData(loss_val)
      end
      current_loss /= length(dataloader)

      step_scheduler!(lr_scheduler,train_state.optimizer_state,epoch,current_loss;verbose=logger.verbose)

      update!(logger,epoch,current_loss)
    end
  end

  finalise!(logger)
  return NeuralOperator(train_state,stats)
end