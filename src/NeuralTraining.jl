build_model(::NeuralModel,args...) = @abstractmethod

function build_lux_chain(layers::Tuple,activation)
  l = ()
  for i in 2:(length(layers)-1)
    l = (l...,Lux.Dense(layers[i-1] => layers[i],activation))
  end
  Lux.Chain(l...,Lux.Dense(layers[end-1] => layers[end]))
end

function build_model(model::KernelNeuralModel,args...)
  #TODO
end

function build_model(model::NOMAD,args...)
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

function build_model(model::MultiLayerPerceptron,(values,coords,params))
  m = size(params,1)
  n = size(values,1)
  build_lux_chain((m,model.hidden_layers...,n),model.activation)
end

function train!(red::KernelReduction,train_state,dataloader,stats)
  lr_scheduler = get_scheduler(red)
  logger = get_logger(red)
  function to_device_batch((x_batch,y_batch))
    #TODO check if this is correct
    return (x_batch |> XDEV,y_batch |> XDEV)
  end
  train_model!(train_state,dataloader,stats,lr_scheduler,to_device_batch;logger)
end

function get_train_state(red::NeuralReduction,inputs;rng=Random.default_rng(),seed=1234)
  Random.seed!(rng,seed)
  model = build_model(red,inputs)
  opt = get_optimiser(red)
  ps,st = Lux.setup(rng,model) |> XDEV
  Lux.Training.TrainState(model,ps,st,opt)
end

function get_data_loader(red::NeuralReduction,inputs;shuffle=true,partial=false)
  batchsize = resolve_batch_size(red,inputs)
  data = prepare_data(red,inputs)
  MLUtils.DataLoader(data;batchsize,shuffle,partial)
end

function train(red::NeuralReduction,inputs,stats;kwargs...)
  train_state = get_train_state(red,inputs;kwargs...)
  dataloader,args... = get_data_loader(red,inputs;shuffle=true,partial=false)
  train!(red,train_state,dataloader,stats,args...)
end

function train(solver::NeuralSolver,inputs,stats;kwargs...)
  train(get_reduction(solver),inputs,stats;kwargs...)
end

function train(
  solver::NeuralSolver,
  feop::ParamOperator,
  data...;
  normalise=true,
  kwargs...
  )

  inputs,stats = get_inputs_and_stats(solver,feop,data...;normalise)
  train(solver,inputs,stats;kwargs...)
end

function train(
  solver::NeuralSolver,
  op::NeuralOperator,
  data...;
  normalise=true,
  update_stats=false,
  kwargs...
  )

  inputs,stats = if update_stats
    get_inputs_and_stats(solver,op,data...;normalise)
  else
    inputs = get_inputs(solver,op,data...)
    stats = get_stats(op)
    normalise && normalise!(inputs,stats)
    inputs,stats
  end
  train(solver,inputs,stats;kwargs...)
end

"""
    train_model!(train_state,dataloader,stats,lr_scheduler,to_device_batch;loss=Lux.MSELoss(),logger) -> TrainedNeuralModel

Generic Lux/Reactant/Enzyme training loop shared by every architecture in this package.
`to_device_batch` maps one raw batch yielded by `dataloader` to whatever `loss` expects
as its data argument (already moved to `XDEV`) — DeepONet pairs each batch with a fixed
set of trunk query points, NOMAD's coordinates are already part of the per-row batch,
and a plain reconstruction network (AutoEncoder/AutoDecoder) just needs `(x,x)`. `loss`
defaults to `Lux.MSELoss()`; pass a custom `(model,ps,st,data) -> (loss,st,stats)`
function for anything else (e.g. a VAE's reconstruction+KL loss). `stats` is the
normalisation bundled into the returned [`TrainedNeuralModel`](@ref).
"""
function train_model!(train_state,dataloader,stats,lr_scheduler,to_device_batch;loss=Lux.MSELoss(),logger)
  init!(logger)

  Reactant.with_config(;dot_general_precision=Reactant.PrecisionConfig.HIGH) do
    for epoch in 1:logger.max_epochs
      local current_loss = 0.0f0

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
  return TrainedNeuralModel(train_state,stats)
end