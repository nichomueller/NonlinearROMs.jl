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
  activation
  )

  c_dims = _coord_dims(style)
  inner_dims = _get_inner_dims(style,formatter,fixed_dims,in_dims,c_dims)
  k_out = (channel_wise && in_dims == out_dims) ? out_dims : out_dims*in_dims
  inner = build_lux_chain((inner_dims,kernel_layers...,k_out),activation)
  kernel = IntegralKernel(inner,style.measure,formatter,c_dims)
  KernelNeuralLayer(kernel,in_dims,out_dims,activation)
end

function build_kernel_neural_layer(
  style::FourierStyle,
  formatter::DataFormatter,
  fixed_dims::Int,in_dims::Int,out_dims::Int,
  kernel_layers::Tuple,
  channel_wise::Bool,
  activation
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
  build_model(s.model,data,get_formatter(s))
end

function build_model(
  model::KernelNeuralModel,
  (inputs,outputs),
  fixed_formatter::DataFormatter=DataFormatter(model)
  )

  in_dims = size(inputs,1)
  out_dims = size(outputs,1)
  widths = (in_dims,model.hidden_layers...,out_dims)
  channels = get_kernel_channels(model.formatter,fixed_formatter,_num_coord_dims(model.style),in_dims)
  lift = Lux.Dense(widths[1] => widths[2],model.activation)
  klayers = ntuple(
    i -> build_kernel_neural_layer(
      model.style,
      model.formatter,
      channels,
      in_dims,
      widths[i+1],
      widths[i+2],
      model.kernel_layers,
      model.channel_wise,
      model.activation
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

function get_data_loader(solver::NeuralSolver,data;kwargs...)
  batchsize = resolve_batch_size(solver,data)
  MLUtils.DataLoader(data;batchsize,kwargs...)
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

# utils

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