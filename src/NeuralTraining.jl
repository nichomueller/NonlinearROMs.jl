const CDEV = Lux.cpu_device()
const XDEV = Lux.reactant_device(;force=true)

function build_lux_chain(layers::Tuple,activation)
  l = ()
  for i in 2:(length(layers)-1)
    l = (l...,Lux.Dense(layers[i-1] => layers[i],activation))
  end
  Lux.Chain(l...,Lux.Dense(layers[end-1] => layers[end]))
end

build_model(::NeuralModel,args...) = @abstractmethod

_coord_dims(::KernelStyle) = 0
function _coord_dims(style::IntegralStyle)
  coords = get_coords(get_space(style.measure))
  isempty(coords) ? 0 : num_components(eltype(coords))
end

function build_kernel_neural_layer(
  style::IntegralStyle,
  formatter::DataFormatter,
  fixed_dims::Int,in_dims::Int,out_dims::Int,
  kernel_layers::Tuple,
  channel_wise::Bool,
  activation,
  kernel_activation
  )

  c_dims = _coord_dims(style)
  inner_dims = _get_inner_dims(style,formatter,fixed_dims,in_dims,c_dims)
  k_out = (channel_wise && in_dims == out_dims) ? out_dims : out_dims*in_dims
  inner = build_lux_chain((inner_dims,kernel_layers...,k_out),kernel_activation)
  kernel = IntegralKernel(inner,style.measure,formatter,c_dims)
  KernelNeuralLayer(kernel,in_dims,out_dims,activation)
end

function build_kernel_neural_layer(
  style::FourierStyle,
  formatter::DataFormatter,
  fixed_dims::Int,in_dims::Int,out_dims::Int,
  kernel_layers::Tuple,
  channel_wise::Bool,
  activation,
  kernel_activation
  )

  inner_dims = _get_inner_dims(style,formatter,fixed_dims,in_dims,0)
  kernel = FourierKernel(inner_dims,out_dims,style.truncation)
  KernelNeuralLayer(kernel,in_dims,out_dims,activation)
end

build_kernel_network(in_dims,out_dims,::Tuple{},activation) = Lux.Dense(in_dims => out_dims)

function build_kernel_network(in_dims,out_dims,layers::Tuple,activation)
  build_lux_chain((in_dims,layers...,out_dims),activation)
end

function build_model(s::NeuralSolver{<:KernelNeuralModel},data)
  build_model(s.model,data)
end

function build_model(model::KernelNeuralModel,(inputs,outputs))
  in_dims = _in_dims(inputs,outputs)
  out_dims = dimension(outputs)
  widths = (in_dims,model.hidden_layers...,out_dims)
  lift = Lux.Dense(widths[1] => widths[2],model.activation)
  klayers = ntuple(
    i -> build_kernel_neural_layer(
      model.style,
      model.formatter,
      in_dims,
      widths[i+1],
      widths[i+2],
      model.kernel_layers,
      model.channel_wise,
      model.activation,
      model.kernel_activation
    ),
    length(widths)-3
  )
  last_h = widths[end-1]
  project = build_lux_chain((last_h,last_h,out_dims),model.activation)
  Lux.Chain(Lift(lift),klayers...,Project(project))
end

function build_model(model::NOMAD,(inputs,outputs))
  Lux.Chain(
    # Apply approximator to 'u', pass 'y' untouched, and concatenate them (vcat)
    Lux.Parallel(
      vcat;
      approximator = build_lux_chain(model.approximator_layers,model.activation),
      y_pass_through = Lux.NoOpLayer()
    ),
    # Pass the concatenated vector [approximator(u); y] to the decoder
    build_lux_chain(model.decoder_layers,model.activation)
  )
end

function build_model(model::MultiLayerPerceptron,(inputs,outputs))
  in_dims = size(inputs,1)
  out_dims = size(outputs,1)
  build_lux_chain((in_dims,model.hidden_layers...,out_dims),model.activation)
end

function train!(solver::NeuralSolver,train_state,dataloader,stats)
  lr_scheduler = get_scheduler(solver)
  loss = get_loss(solver)
  logger = get_logger(solver)
  to_device_batch((x_batch,y_batch)) = (x_batch |> XDEV,y_batch |> XDEV)
  train_model!(train_state,dataloader,stats,lr_scheduler,loss,to_device_batch;logger)
end

function get_train_state(solver::NeuralSolver,data;rng=Random.default_rng(),seed=1234)
  Random.seed!(rng,seed)
  model = build_model(solver,data)
  opt = get_optimiser(solver)
  ps,st = Lux.setup(rng,model) |> XDEV
  Lux.Training.TrainState(model,ps,st,opt)
end

resolve_data_loader(a) = MLUtils.DataLoader
resolve_data_loader(a::TwoStageBuckets) = AutoregressiveDataLoader
resolve_data_loader((input,output)) = resolve_data_loader(input)

function get_data_loader(solver::NeuralSolver,data;kwargs...)
  batchsize = resolve_batch_size(solver,data)
  dloader = resolve_data_loader(data)
  dloader(data;batchsize,kwargs...)
end

function train(solver::NeuralSolver,data,stats;rng=Random.default_rng(),seed=1234,kwargs...)
  train_state = get_train_state(solver,data;rng,seed)
  dataloader = get_data_loader(solver,data;kwargs...)
  train!(solver,train_state,dataloader,stats)
end

"""
    train_model!(train_state,dataloader,stats,lr_scheduler,loss,to_device_batch;logger) -> NeuralOperator

Generic Lux/Reactant/Enzyme training loop shared by every architecture in this package.
`to_device_batch` maps one raw batch yielded by `dataloader` to whatever `loss` expects
as its data argument (already moved to `XDEV`) — DeepONet pairs each batch with a fixed
set of trunk query points, NOMAD's coordinates are already part of the per-row batch,
and a plain reconstruction network (AutoEncoder/AutoDecoder) just needs `(x,x)`. `loss`
defaults to `Lux.MSELoss()`; pass a custom `(model,ps,st,data) -> (loss,st,stats)`
function for anything else (e.g. a VAE's reconstruction+KL loss). `stats` is the
normalisation bundled into the returned [`NeuralOperator`](@ref).
"""
function train_model!(train_state,dataloader,stats,lr_scheduler,loss,to_device_batch;logger)
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

# rollout 

function AutoregressiveDataLoader((input,output);kwargs...)
  one_step_data,pushforward_data = rollout_data(input,output)
  (
    one_step = MLUtils.DataLoader(one_step_data;kwargs...),
    pushforward = MLUtils.DataLoader(pushforward_data;kwargs...)
  )
end

function rollout_data(input::TwoStageBuckets,output::TwoStageBuckets)
  nc = input.num_clusters
  @check output.num_clusters == nc "Input and output must have the same number of clusters"
  one_step_data = (
    input = _merge_data(input.second_stage,output.first_stage),
    output = output.second_stage
  )
  
  in_first_forward = _get_clusters(input.first_stage,2:nc,nc)
  in_back = _get_clusters(one_step_data.input,1:(nc-1),nc)
  out_first_back = _get_clusters(output.first_stage,1:(nc-1),nc)
  pushforward_data = (
    input = in_first_forward,
    output = out_first_back,
    metadata = in_back
  )
  return (one_step_data,pushforward_data)
end

function train_model!(
  train_state,
  (one_step,pushforward)::NamedTuple,
  stats,
  lr_scheduler,
  loss,
  to_device_batch;
  logger,λ=TData(1),warmup_epochs=0
  )

  # one-step term + pushforward term; ŷp is passed as data, so it is constant
  # for Enzyme (stop-gradient)
  function rollout_loss(model,ps,st,(one_step,pushforward,pred))
    updated_pushforward = (
      input = _merge_data(pushforward.input,pred),
      output = pushforward.output
    )

    l_one_step,st,_ = loss(model,ps,st,one_step)
    ŷ,st = model(updated_pushforward,ps,st)
    l_pushforward = loss(ŷ,pushforward.output)
    return l_one_step + TData(λ)*l_pushforward,st,(;)
  end

  init!(logger)

  predict = nothing

  Reactant.with_config(;dot_general_precision=Reactant.PrecisionConfig.HIGH) do
    for epoch in 1:logger.max_epochs
      local current_loss = zero(TData)

      for raw_batch in loader
        raw_one_step,raw_pushforward = raw_batch
        one_step_batch = to_device_batch(raw_one_step)
        pushforward_batch = to_device_batch(raw_pushforward)

        if epoch ≤ warmup_epochs
          _,loss_val,_,train_state = Lux.Training.single_train_step!(
            Lux.AutoEnzyme(),
            loss,
            one_step_batch,
            train_state;
            return_gradients=Val(false)
          )
        else
          ps = train_state.parameters
          st = Lux.testmode(train_state.states)
          if isnothing(predict)
            predict = Reactant.@compile train_state.model(pushforward_batch.metadata,ps,st)
          end
          # plain forward pass outside the differentiated call
          ŷ, = predict(pushforward_batch.metadata,ps,st)
          _,loss_val,_,train_state = Lux.Training.single_train_step!(
            Lux.AutoEnzyme(),
            rollout_loss,
            (one_step_batch,pushforward_batch,ŷ),
            train_state;
            return_gradients=Val(false)
          )
        end

        current_loss += TData(loss_val)
      end
      current_loss /= length(loader)

      step_scheduler!(lr_scheduler,train_state.optimizer_state,epoch,current_loss;verbose=logger.verbose)

      update!(logger,epoch,current_loss)
    end
  end

  finalise!(logger)
  return NeuralOperator(train_state,stats)
end

# utils

_in_dims(inputs,outputs) = dimension(inputs)
# rollout: the previous state is appended to the fixed data
_in_dims(inputs::TwoStageBuckets,outputs::TwoStageBuckets) = dimension(inputs) + dimension(outputs)

_shape_factor(::KernelStyle) = 1
_edge_dims(::KernelStyle) = 0

_shape_factor(s::IntegralStyle) = _shape_factor(get_message_rule(s.measure))
_edge_dims(s::IntegralStyle) = _edge_dims(get_message_rule(s.measure))

_shape_factor(::MPRule) = 1
_shape_factor(::PlainMP) = 2

_edge_dims(::MPRule) = 0
_edge_dims(::PlainMP) = 1
_edge_dims(::CompositeMP) = 1

function _selected_fixed_dims(::DataFormatter{A,B},fixed_dims::Int,coord_dims::Int) where {A,B}
  if coord_dims > 0 && fixed_dims > coord_dims
    (A ? coord_dims : 0) + (B ? (fixed_dims - coord_dims) : 0)
  else
    (A || B) ? fixed_dims : 0
  end
end

_augment_dims(::DataFormatter{<:Any,<:Any,C}) where C = C

function _get_inner_dims(s,d,fixed_dims,in_dims,coord_dims=0)
  sel_fixed = _selected_fixed_dims(d,fixed_dims,coord_dims)
  _shape_factor(s) * (sel_fixed + _augment_dims(d)*in_dims) + _edge_dims(s)
end