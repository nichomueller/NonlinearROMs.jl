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

_pass_new_data(::DataFormatter{<:Any,<:Any,C}) where C = C

function build_kernel_neural_layer(
  style::IntegralStyle,
  formatter::DataFormatter,
  din::Int,in_dims::Int,out_dims::Int,
  activation
  )

  merged = _pass_new_data(formatter) ? din+in_dims : din
  inner = Lux.Dense(merged*kernel_input_multiplier(style.measure) => out_dims)
  kernel = IntegralKernel(inner,style.measure,formatter)
  KernelNeuralLayer(kernel,in_dims,out_dims,activation)
end

function build_kernel_neural_layer(
  style::FourierStyle,
  formatter::DataFormatter,
  din::Int,in_dims::Int,out_dims::Int,
  activation
  )

  merged = _pass_new_data(formatter) ? din+in_dims : din
  kernel = FourierKernel(merged,out_dims,style.truncation)
  KernelNeuralLayer(kernel,in_dims,out_dims,activation)
end

function build_model(model::KernelNeuralModel,(inputs,outputs))
  input_channels = size(inputs,1)
  output_channels = size(outputs,1)
  widths = (input_channels,model.hidden_layers...,output_channels)
  lift = Lux.Dense(widths[1] => widths[2],model.activation)
  klayers = ntuple(
    i -> build_kernel_neural_layer(
      model.style,
      model.formatter,
      input_channels,
      widths[i+1],
      widths[i+2],
      model.activation
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
  input_channels = size(inputs,1)
  output_channels = size(outputs,1)
  build_lux_chain((input_channels,model.hidden_layers...,output_channels),model.activation)
end

function train!(red::KernelReduction,train_state,dataloader,stats)
  lr_scheduler = get_scheduler(red)
  logger = get_logger(red)
  function to_device_batch((x_batch,y_batch))
    (x_batch |> XDEV,y_batch |> XDEV)
  end
  train_model!(train_state,dataloader,stats,lr_scheduler,to_device_batch;logger)
end

function get_train_state(red::NeuralReduction,data;rng=Random.default_rng(),seed=1234)
  Random.seed!(rng,seed)
  model = build_model(red,data)
  opt = get_optimiser(red)
  ps,st = Lux.setup(rng,model) |> XDEV
  Lux.Training.TrainState(model,ps,st,opt)
end

function get_data_loader(red::NeuralReduction,data;shuffle=true,partial=false)
  batchsize = resolve_batch_size(red,data)
  MLUtils.DataLoader(data;batchsize,shuffle,partial)
end

function train(red::NeuralReduction,data,stats;kwargs...)
  train_state = get_train_state(red,data;kwargs...)
  dataloader = get_data_loader(red,data;shuffle=true,partial=false)
  train!(red,train_state,dataloader,stats)
end

function train(solver::NeuralSolver,data,stats;kwargs...)
  train(get_reduction(solver),data,stats;kwargs...)
end

"""
    train_model!(train_state,dataloader,stats,lr_scheduler,to_device_batch;loss=Lux.MSELoss(),logger) -> NeuralOperator

Generic Lux/Reactant/Enzyme training loop shared by every architecture in this package.
`to_device_batch` maps one raw batch yielded by `dataloader` to whatever `loss` expects
as its data argument (already moved to `XDEV`) — DeepONet pairs each batch with a fixed
set of trunk query points, NOMAD's coordinates are already part of the per-row batch,
and a plain reconstruction network (AutoEncoder/AutoDecoder) just needs `(x,x)`. `loss`
defaults to `Lux.MSELoss()`; pass a custom `(model,ps,st,data) -> (loss,st,stats)`
function for anything else (e.g. a VAE's reconstruction+KL loss). `stats` is the
normalisation bundled into the returned [`NeuralOperator`](@ref).
"""
function train_model!(train_state,dataloader,stats,lr_scheduler,to_device_batch;loss=Lux.MSELoss(),logger)
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