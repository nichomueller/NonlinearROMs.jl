function train!(solver::NeuralSolver{<:KernelReduction},train_state,dataloader,stats)
  lr_scheduler = get_scheduler(solver)
  logger = get_logger(solver)
  function to_device_batch((x_batch,y_batch))
    n_dofs,n_samples = size(y_batch)
    # Dynamically compute the number of physical variables
    out_channels = n_dofs ÷ n_nodes
    y_reshaped = reshape(y_batch,out_channels,n_nodes,n_samples)
    return (x_batch |> XDEV,y_reshaped |> XDEV)
  end
  train_model!(train_state,dataloader,stats,lr_scheduler,to_device_batch;logger)
end

function train!(solver::NeuralSolver{<:DeepONetReduction},train_state,dataloader,stats,x_data_dev)
  lr_scheduler = get_scheduler(solver)
  logger = get_logger(solver)
  to_device_batch((f_batch,u_batch)) = ((f_batch |> XDEV,x_data_dev),u_batch |> XDEV)
  train_model!(train_state,dataloader,stats,lr_scheduler,to_device_batch;logger)
end

function train!(solver::NeuralSolver{<:NOMADReduction},train_state,dataloader,stats)
  lr_scheduler = get_scheduler(solver)
  logger = get_logger(solver)
  to_device_batch(((u_batch,y_batch),v_batch)) = ((u_batch |> XDEV,y_batch |> XDEV),v_batch |> XDEV)
  train_model!(train_state,dataloader,stats,lr_scheduler,to_device_batch;logger)
end

function train!(solver::NeuralSolver{<:DeepONetReduction},train_state,dataloader,stats)
  lr_scheduler = get_scheduler(solver)
  logger = get_logger(solver)
  to_device_batch((xb,yb)) = (xb |> XDEV,yb |> XDEV)
  train_model!(train_state,dataloader,stats,lr_scheduler,to_device_batch;logger)
end

function train!(solver::NeuralSolver{<:AutoEncoderReduction},train_state,dataloader,stats)
  lr_scheduler = get_scheduler(solver)
  logger = get_logger(solver)
  to_device_batch((xb,yb)) = (xb |> XDEV,yb |> XDEV)
  train_model!(train_state,dataloader,stats,lr_scheduler,to_device_batch;logger)
end

function train!(solver::NeuralSolver{<:VAEReduction},train_state,dataloader,stats)
  red = get_reduction(solver)
  lr_scheduler = get_scheduler(solver)
  logger = get_logger(solver)
  β = red.model.β
  loss(model,ps,st,x) = vae_loss(model,ps,st,x;β)
  to_device_batch(xb) = xb |> XDEV
  train_model!(train_state,dataloader,stats,lr_scheduler,to_device_batch;loss,logger)
end

function train!(solver::NeuralSolver{<:MultiLayerPerceptron},train_state,dataloader,stats)
  lr_scheduler = get_scheduler(solver)
  logger = get_logger(solver)
  to_device_batch((x,y)) = (x |> XDEV,y |> XDEV)
  train_model!(train_state,dataloader,stats,lr_scheduler,to_device_batch;logger)
end

function get_train_state(solver::NeuralSolver,inputs;rng=Random.default_rng(),seed=1234)
  Random.seed!(rng,seed)
  model = build_model(solver,inputs)
  opt = get_optimiser(solver)
  ps,st = Lux.setup(rng,model) |> XDEV
  Lux.Training.TrainState(model,ps,st,opt)
end

function get_data_loader(solver::NeuralSolver,inputs;shuffle=true,partial=false)
  batchsize = resolve_batch_size(solver,inputs)
  data = prepare_data(solver,inputs)
  MLUtils.DataLoader(data;batchsize,shuffle,partial)
end

function get_data_loader(solver::NeuralSolver{<:DeepONetReduction},inputs;shuffle=true,partial=false)
  batchsize = resolve_batch_size(solver,inputs)
  data = prepare_data(solver,inputs)
  dataloader = MLUtils.DataLoader(data;batchsize,shuffle,partial)
  coords_dev = coords |> XDEV
  (dataloader,coords_dev)
end

function train(solver::NeuralSolver,inputs,stats;kwargs...)
  train_state = get_train_state(solver,inputs;kwargs...)
  dataloader,args... = get_data_loader(solver,inputs;shuffle=true,partial=false)
  train!(solver,train_state,dataloader,stats,args...)
end

function train(solver::NeuralSolver{<:AutoDecoderReduction},inputs,stats;kwargs...)
  train_state = get_train_state(solver,inputs;kwargs...)
  dataloader,args... = get_data_loader(solver,inputs;shuffle=false,partial=false)
  train!(solver,train_state,dataloader,stats,args...)
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
  update_stats=false,
  normalise=true,
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

# utils

function prepare_data(::NeuralSolver{<:KernelReduction},(values,coords,params))
  (values,tensor_of_coords(coords,params))
end

function prepare_data(::NeuralSolver{<:DeepONetReduction},(values,coords,params))
  (params,values)
end

function prepare_data(::NeuralSolver{<:NOMADReduction},(values,coords,params))
  ntot = size(coords,2)*size(params,2)
  p = zeros(eltype(params),size(params,1),ntot)
  x = zeros(eltype(coords),size(coords,1),ntot)
  d = zeros(eltype(values),1,ntot)

  col = 1
  @views for i in axes(params,2)
    p = params[:,i]
    for j in axes(coords,2)
      p[:,col] = p
      x[:,col] = coords[:,j]
      d[1,col] = data[j,i]
      col += 1
    end
  end

  (p,x),d
end

for T in (:AutoEncoderReduction,:AutoDecoderReduction,:VAEReduction,)
  @eval begin
    function prepare_data(::NeuralSolver{<:$T},(values,coords,params))
      (values,values)
    end
  end
end

function prepare_data(::NeuralSolver{<:MLPReduction},(values,coords,params))
  (params,values)
end