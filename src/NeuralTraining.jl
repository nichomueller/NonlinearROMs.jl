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

function build_kernel_neural_layer(
  style::IntegralStyle,
  formatter::DataFormatter,
  channels,fixed_dims::Int,in_dims::Int,out_dims::Int,
  activation,kernel_layers::Tuple,kernel_activation
  )

  @check !isnothing(channels) || _augment_dims(formatter) "The kernel has no input: pass coords, features or new data"
  inner_dims = _get_inner_dims(style,formatter,_num_channels(channels,fixed_dims),in_dims)
  inner = build_kernel_network(inner_dims,out_dims*in_dims,kernel_layers,kernel_activation)
  kernel = IntegralKernel(inner,style.measure,formatter,channels)
  KernelNeuralLayer(kernel,in_dims,out_dims,activation)
end

function build_kernel_neural_layer(
  style::FourierStyle,
  formatter::DataFormatter,
  channels,fixed_dims::Int,in_dims::Int,out_dims::Int,
  activation,args...
  )

  inner_dims = _get_inner_dims(style,formatter,fixed_dims,in_dims)
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
      model.activation,
      model.kernel_layers,
      model.kernel_activation
    ),
    length(widths)-3
  )
  project = Lux.Dense(widths[end-1] => widths[end])
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
_shape_factor(::IntegralStyle{<:GraphMeasure}) = 2

_augment_dims(::DataFormatter{<:Any,<:Any,C}) where C = C

_num_coord_dims(::KernelStyle) = 0
_num_coord_dims(s::IntegralStyle) = num_components(eltype(get_coords(get_space(s.measure))))

_num_channels(::Colon,fixed_dims) = fixed_dims
_num_channels(::Nothing,fixed_dims) = 0
_num_channels(ids::AbstractUnitRange,fixed_dims) = length(ids)

function _get_inner_dims(s,d,fixed_dims,in_dims)
  _shape_factor(s) * (fixed_dims + _augment_dims(d)*in_dims)
end