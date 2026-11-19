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
using Random
rng=Random.default_rng()
seed=1234
train_state = NonlinearROMs.get_train_state(solver,data;rng,seed)
dataloader = NonlinearROMs.get_data_loader(solver,data)

function rollout_dataset(input::TwoStageBuckets,output::TwoStageBuckets)
  n = length(output.first_stage)
  X1 = [_merge_data(input.second_stage[k],output.first_stage[k]) for k in 1:n]
  Xp = [X1[max(k-1,1)] for k in 1:n]  # cluster k-1's one-step input (unused for k = 1)
  w = [fill(TData(k > 1),1,1,num_samples(output.first_stage[k])) for k in 1:n]
  c(a) = cat(a...;dims=3)
  (
    X1 = c(X1),                     # one-step input
    Y1 = c(output.second_stage),    # one-step target
    Xp = c(Xp),                     # input whose prediction gets pushed forward
    F = c(input.first_stage),       # fixed data at the pushforward target times
    Ys = c(output.first_stage),     # pushforward target
    w = c(w)                        # 0 masks the pushforward term for k = 1
  )
end


function train_model!(
  train_state,
  dataloader::MLUtils.DataLoader{<:NamedTuple},
  stats,
  lr_scheduler,
  loss,
  to_device_batch;
  logger,λ=1.0,warmup_epochs=0
  )

  init!(logger)

  # one-step term with gradient; pushforward term on a detached prediction ŷ
  function rollout_loss(model,ps,st,(X1,Y1,F,ŷ,Ys,w))
    l1,st,_ = loss(model,ps,st,(X1,Y1))
    ŷs,st = model(_merge_data(F,ŷ),ps,st)
    l2 = loss(ŷs .* w,Ys .* w)
    return l1 + TData(λ)*l2,st,(;)
  end

  local predict = nothing

  Reactant.with_config(;dot_general_precision=Reactant.PrecisionConfig.HIGH) do
    for epoch in 1:logger.max_epochs
      local current_loss = zero(TData)

      for raw_batch in dataloader
        (;X1,Y1,Xp,F,Ys,w) = to_device_batch(raw_batch)

        if epoch <= warmup_epochs
          _,loss_val,_,train_state = Lux.Training.single_train_step!(
            Lux.AutoEnzyme(),loss,(X1,Y1),train_state;return_gradients=Val(false)
          )
        else
          ps,st = train_state.parameters,Lux.testmode(train_state.states)
          isnothing(predict) && (predict = Reactant.@compile train_state.model(Xp,ps,st))
          # plain forward outside the differentiated call: ŷ enters the loss as data
          ŷ, = predict(Xp,ps,st)
          _,loss_val,_,train_state = Lux.Training.single_train_step!(
            Lux.AutoEnzyme(),rollout_loss,(X1,Y1,F,ŷ,Ys,w),train_state;return_gradients=Val(false)
          )
        end
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
