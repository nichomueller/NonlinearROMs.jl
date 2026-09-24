function train_kernel_operator!(train_state,dataloader,red,n_nodes)
  lr_scheduler = get_scheduler(red)
  logger = get_logger(red)

  # prepares the mini-batch for the neural network.
  # x_batch is [Features_in, Nodes, Samples].
  # y_batch is [N_dofs, Samples]. It gets reshaped to [Features_out, Nodes, Samples].
  function to_device_batch((x_batch,y_batch))
    n_samples = size(y_batch,2)
    n_dofs = size(y_batch,1)

    # Dynamically compute the number of physical variables
    out_channels = n_dofs ÷ n_nodes

    y_reshaped = reshape(y_batch,out_channels,n_nodes,n_samples)

    return (x_batch |> XDEV,y_reshaped |> XDEV)
  end

  train_model!(train_state,dataloader,lr_scheduler,to_device_batch;logger)
end

function train(
  red::KernelReduction,
  feop::ParamOperator,
  s::AbstractSnapshots
  )

  # Data extraction
  sx = CoordinateSnapshots(s,get_test(feop))
  target = sample(red,sx)
  data,params,coords = get_formatted_data(Float32,target)

  # Normalisation applied strictly before building the tensor
  stats = NormStats(data,params,coords;normalise=true)

  # Build 3D Tensor for Kernel Operators
  input_tensor = tensor_of_coords(coords,params)
  n_samples = size(data,2)

  # Model Building
  rng = Random.default_rng()
  Random.seed!(rng,42)

  model = build_model(red)
  opt = get_optimiser(red)
  ps,st = Lux.setup(rng,model) |> XDEV
  train_state = Lux.Training.TrainState(model,ps,st,opt)

  # DataLoader
  bs = resolve_batch_size(red,n_samples)
  dataloader = MLUtils.DataLoader(
    (input_tensor,data);
    batchsize=bs,
    shuffle=true,
    partial=false
  )

  trained = train_kernel_operator!(train_state,dataloader,red)

  return trained,stats
end

function train(
  red::KernelReduction,
  feop::ParamOperator,
  s::AbstractSnapshots,
  pretrained_op::NeuralOperator;
  update_stats::Bool=false
  )

  # Data extraction
  sx = CoordinateSnapshots(s,get_test(feop))
  target = sample(red,sx)
  data,params,coords = get_formatted_data(Float32,target)
  n_samples = size(data,2)

  # Normalisation
  if update_stats
    stats = NormStats(data,params,coords;normalise=true)
  else
    stats = pretrained_op.metadata
    expected_param_in = length(stats.pscore.μ)
    expected_coord_in = length(stats.xscore.μ)
    @assert size(params,1) == expected_param_in "Parameter dimension mismatch: expected $expected_param_in, got $(size(params,1))."
    @assert size(coords,1) == expected_coord_in "Coordinate dimension mismatch: expected $expected_coord_in, got $(size(coords,1))."
    normalise!((data,params,coords),stats)
  end

  # Build 3D Tensor for Kernel Operators
  input_tensor = tensor_of_coords(coords,params)

  # Pretrained model setup
  model = pretrained_op.model.chain
  opt = get_optimiser(red)
  ps = pretrained_op.model.parameters |> XDEV
  st = pretrained_op.model.states |> XDEV
  train_state = Lux.Training.TrainState(model,ps,st,opt)

  bs = resolve_batch_size(red,n_samples)
  dataloader = MLUtils.DataLoader(
    (input_tensor,data);
    batchsize=bs,
    shuffle=true,
    partial=false
  )

  trained = train_kernel_operator!(train_state,dataloader,red)

  return trained,stats
end

function train_deeponet!(train_state,dataloader,x_data_dev,red)
  lr_scheduler = get_scheduler(red)
  logger = get_logger(red)
  to_device_batch((f_batch,u_batch)) = ((f_batch |> XDEV,x_data_dev),u_batch |> XDEV)
  train_model!(train_state,dataloader,lr_scheduler,to_device_batch;logger)
end

function train_nomad!(train_state,dataloader,red)
  lr_scheduler = get_scheduler(red)
  logger = get_logger(red)
  to_device_batch(((u_batch,y_batch),v_batch)) = ((u_batch |> XDEV,y_batch |> XDEV),v_batch |> XDEV)
  train_model!(train_state,dataloader,lr_scheduler,to_device_batch;logger)
end

function train_autoencoder!(train_state,dataloader,red)
  lr_scheduler = get_scheduler(red)
  logger = get_logger(red)
  to_device_batch((xb,yb)) = (xb |> XDEV,yb |> XDEV)
  train_model!(train_state,dataloader,lr_scheduler,to_device_batch;logger)
end

function train_autodecoder!(train_state,dataloader,red)
  lr_scheduler = get_scheduler(red)
  logger = get_logger(red)
  to_device_batch((xb,yb)) = (xb |> XDEV,yb |> XDEV)
  train_model!(train_state,dataloader,lr_scheduler,to_device_batch;logger)
end

function train_vae!(train_state,dataloader,red)
  lr_scheduler = get_scheduler(red)
  logger = get_logger(red)
  β = red.model.β
  function vae_loss(model::VAELayer,ps,st,x)
    n_h = size(x,1)
    out,st = model(x,ps,st)
    x̂ = view(out,1:n_h,:)
    μ = view(out,n_h+1:n_h+model.latent_dim,:)
    log_var = view(out,n_h+model.latent_dim+1:size(out,1),:)
    recon = sum(abs2,x̂ .- x)/length(x)
    kl = -sum(1 .+ log_var .- μ.^2 .- exp.(log_var))/(2*size(x,2))
    return recon + β*kl,st,(;)
  end
  to_device_batch(xb) = xb |> XDEV
  train_model!(train_state,dataloader,lr_scheduler,to_device_batch;loss=vae_loss,logger)
end

# Generic Dispatch (Steady)

function train(
  red::DeepONetReduction,
  feop::ParamOperator,
  s::AbstractSnapshots
  )

  # Data extraction
  sx = CoordinateSnapshots(s,get_test(feop))
  target = sample(red,sx)
  data,params,coords = get_formatted_data(Float32,target)

  # Normalisation
  stats = NormStats(data,params,coords;normalise=true)

  # Building the DeepONet
  rng = Random.default_rng()
  Random.seed!(rng,42)

  model = build_model(red)
  opt = get_optimiser(red)
  coords_dev = coords |> XDEV
  ps,st = Lux.setup(rng,model) |> XDEV
  train_state = Lux.Training.TrainState(model,ps,st,opt)

  # Dataloader and setup
  bs = resolve_batch_size(red,num_params(s))
  dataloader = MLUtils.DataLoader(
    (params,data);
    batchsize=bs,
    shuffle=true,
    partial=false
  )

  # Executing the pipeline
  trained = train_deeponet!(train_state,dataloader,coords_dev,red)

  return trained,stats
end

function train(
  red::DeepONetReduction,
  feop::ParamOperator,
  s::AbstractSnapshots,
  pretrained_op::NeuralOperator;
  update_stats::Bool=false
  )

  # Data extraction
  sx = CoordinateSnapshots(s,get_test(feop))
  target = sample(red,sx)
  data,params,coords = get_formatted_data(Float32,target)

  # Normalisation
  if update_stats
    stats = NormStats(data,params,coords;normalise=true)
  else
    stats = pretrained_op.metadata
    expected_branch_in = length(stats.pscore.μ)
    expected_trunk_in = length(stats.xscore.μ)
    @assert size(params,1) == expected_branch_in "Branch dimension mismatch: expected $expected_branch_in, got $(size(params,1)). Check the parameter sampler."
    @assert size(coords,1) == expected_trunk_in "Trunk dimension mismatch: expected $expected_trunk_in, got $(size(coords,1))."
    normalise!((data,params,coords),stats)
  end

  # Pretrained model
  model = pretrained_op.model.chain
  opt = get_optimiser(red)
  coords_dev = coords |> XDEV
  ps = pretrained_op.model.parameters |> XDEV
  st = pretrained_op.model.states |> XDEV
  train_state = Lux.Training.TrainState(model,ps,st,opt)

  # Dataloader and setup
  bs = resolve_batch_size(red,num_params(s))
  dataloader = MLUtils.DataLoader(
    (params,data);
    batchsize=bs,
    shuffle=true,
    partial=false
  )

  # Executing the pipeline
  trained = train_deeponet!(train_state,dataloader,coords_dev,red)

  return trained,stats
end

function train(
  red::NOMADReduction,
  feop::ParamOperator,
  s::AbstractSnapshots
  )

  # Data extraction
  sx = CoordinateSnapshots(s,get_test(feop))
  target = sample(red,sx)
  data,params,coords = get_formatted_data(Float32,target)
  dout,pin,xin = _flatten(data,params,coords) # Flattening for NOMAD
  N_tot = size(dout,2)

  # Normalisation
  stats = NormStats(dout,pin,xin;normalise=true)

  # Building the NOMAD model
  rng = Random.default_rng()
  Random.seed!(rng,42)

  model = build_model(red)
  opt = get_optimiser(red)
  ps,st = Lux.setup(rng,model) |> XDEV
  train_state = Lux.Training.TrainState(model,ps,st,opt)

  # DataLoader and Lux setup
  bs = resolve_batch_size(red,N_tot)
  dataloader = MLUtils.DataLoader(
    ((pin,xin),dout);
    batchsize=bs,
    shuffle=true,
    partial=false
  )

  # Running the pipeline
  trained = train_nomad!(train_state,dataloader,red)

  return trained,stats
end

function train(
  red::NOMADReduction,
  feop::ParamOperator,
  s::AbstractSnapshots,
  pretrained_op::NeuralOperator;
  update_stats::Bool=false
  )

  # Data extraction
  sx = CoordinateSnapshots(s,get_test(feop))
  target = sample(red,sx)
  data,params,coords = get_formatted_data(Float32,target)
  dout,pin,xin = _flatten(data,params,coords) # Flattening for NOMAD
  N_tot = size(dout,2)

  # Normalisation
  if update_stats
    stats = NormStats(dout,pin,xin;normalise=true)
  else
    stats = pretrained_op.metadata
    expected_sensors = length(stats.pscore.μ)
    expected_coords = length(stats.xscore.μ)
    @assert size(pin,1) == expected_sensors "Sensors input dimension mismatch: expected $expected_sensors, got $(size(pin,1))."
    @assert size(xin,1) == expected_coords "Coords input dimension mismatch: expected $expected_coords, got $(size(xin,1))."
    normalise!((dout,pin,xin),stats)
  end

  # Pretrained model
  model = pretrained_op.model.chain
  opt = get_optimiser(red)
  ps = pretrained_op.model.parameters |> XDEV
  st = pretrained_op.model.states |> XDEV
  train_state = Lux.Training.TrainState(model,ps,st,opt)

  # DataLoader and Lux setup
  bs = resolve_batch_size(red,N_tot)
  dataloader = MLUtils.DataLoader(
    ((pin,xin),dout);
    batchsize=bs,
    shuffle=true,
    partial=false
  )

  # Running the pipeline
  trained = train_nomad!(train_state,dataloader,red)

  return trained,stats
end

function train(
  red::AutoEncoderReduction,
  feop::ParamOperator,
  s::AbstractSnapshots
  )

  # Data extraction
  data,= get_formatted_data(Float32,s)
  n_samples = size(data,2)

  rng = Random.default_rng()
  Random.seed!(rng,42)

  model = build_model(red.model,size(data,1))
  opt = get_optimiser(red)
  ps,st = Lux.setup(rng,model) |> XDEV
  train_state = Lux.Training.TrainState(model,ps,st,opt)

  bs = resolve_batch_size(red,n_samples)
  dataloader = MLUtils.DataLoader((data,data);batchsize=bs,shuffle=true,partial=false)

  trained = train_autoencoder!(train_state,dataloader,red)

  return trained,identity
end

function train(
  red::AutoEncoderReduction,
  feop::ParamOperator,
  s::AbstractSnapshots,
  pretrained_op::NeuralOperator;
  update_stats::Bool=false
  )

  # Data extraction
  data,= get_formatted_data(Float32,s)
  n_samples = size(data,2)

  # Pretrained model
  model = pretrained_op.model.chain
  opt = get_optimiser(red)
  ps = pretrained_op.model.parameters |> XDEV
  st = pretrained_op.model.states |> XDEV
  train_state = Lux.Training.TrainState(model,ps,st,opt)

  bs = resolve_batch_size(red,n_samples)
  dataloader = MLUtils.DataLoader((data,data);batchsize=bs,shuffle=true,partial=false)

  trained = train_autoencoder!(train_state,dataloader,red)

  return trained,identity
end

function train(
  red::AutoDecoderReduction,
  feop::ParamOperator,
  s::AbstractSnapshots
  )

  # Data extraction
  data,= get_formatted_data(Float32,s)
  nin,n_train = size(data,1),size(data,2)

  rng = Random.default_rng()
  Random.seed!(rng,42)

  model = build_model(red.model,nin,n_train)
  opt = get_optimiser(red)
  ps,st = Lux.setup(rng,model) |> XDEV
  train_state = Lux.Training.TrainState(model,ps,st,opt)

  # Joint decoder + latent-code optimisation is inherently full-batch: every
  # column of the latent-code parameter must be updated on every step.
  dataloader = MLUtils.DataLoader((data,data);batchsize=n_train,shuffle=false,partial=false)

  trained = train_autodecoder!(train_state,dataloader,red)

  return trained,identity
end

function train(
  red::AutoDecoderReduction,
  feop::ParamOperator,
  s::AbstractSnapshots,
  pretrained_op::NeuralOperator;
  update_stats::Bool=false
  )

  # Data extraction
  data,= get_formatted_data(Float32,s)
  n_train = size(data,2)

  # Latent codes are optimised per training sample, so they can't be inherited across a
  # different sample set: fine-tuning keeps the pretrained decoder weights but fits fresh
  # latent codes for the new snapshots.
  decoder = pretrained_op.model.chain.layers.layer_2
  latent_dim = size(pretrained_op.model.parameters.layer_1.codes,1)

  rng = Random.default_rng()
  Random.seed!(rng,42)
  Z0 = randn(Float32,latent_dim,n_train) .* 0.01f0
  model = Lux.Chain(LatentCodeLayer(Z0),decoder)

  ps,st = Lux.setup(rng,model)
  ps = (layer_1=ps.layer_1,layer_2=pretrained_op.model.parameters.layer_2) |> XDEV
  st = (layer_1=st.layer_1,layer_2=pretrained_op.model.states.layer_2) |> XDEV
  opt = get_optimiser(red)
  train_state = Lux.Training.TrainState(model,ps,st,opt)

  dataloader = MLUtils.DataLoader((data,data);batchsize=n_train,shuffle=false,partial=false)

  trained = train_autodecoder!(train_state,dataloader,red)

  return trained,identity
end

function train(
  red::VAEReduction,
  feop::ParamOperator,
  s::AbstractSnapshots
  )

  # Data extraction
  data,= get_formatted_data(Float32,s)
  n_samples = size(data,2)

  rng = Random.default_rng()
  Random.seed!(rng,42)

  model = build_model(red.model,size(data,1))
  opt = get_optimiser(red)
  ps,st = Lux.setup(rng,model) |> XDEV
  train_state = Lux.Training.TrainState(model,ps,st,opt)

  bs = resolve_batch_size(red,n_samples)
  dataloader = MLUtils.DataLoader(data;batchsize=bs,shuffle=true,partial=false)

  trained = train_vae!(train_state,dataloader,red)

  return trained,identity
end

function train(
  red::VAEReduction,
  feop::ParamOperator,
  s::AbstractSnapshots,
  pretrained_op::NeuralOperator;
  update_stats::Bool=false
  )

  # Data extraction
  data,= get_formatted_data(Float32,s)
  n_samples = size(data,2)

  # Pretrained model
  model = pretrained_op.model.chain
  opt = get_optimiser(red)
  ps = pretrained_op.model.parameters |> XDEV
  st = pretrained_op.model.states |> XDEV
  train_state = Lux.Training.TrainState(model,ps,st,opt)

  bs = resolve_batch_size(red,n_samples)
  dataloader = MLUtils.DataLoader(data;batchsize=bs,shuffle=true,partial=false)

  trained = train_vae!(train_state,dataloader,red)

  return trained,identity
end

"""
    train_neural_coefficient(red::NeuralReduction,r::AbstractRealisation,coeff) -> NeuralModel

Builds and trains a [`NeuralModel`](@ref) from `red.model`'s recipe and
`(r,coeff)` data, through the same Lux/Reactant/Enzyme pipeline used for DeepONet/NOMAD.
For a [`MultiLayerPerceptron`](@ref), the input/output dimensions are inferred from
`r`/`coeff` and appended to `red.model.hidden_layers`; for an [`AutoEncoder`](@ref),
`r` is ignored and the network is trained to reconstruct `coeff`.
"""
function train_neural_coefficient(red::NeuralReduction{<:MultiLayerPerceptron},r::AbstractRealisation,coeff)
  x = Float32.(matrix_of_params(r))
  y = Float32.(_get_data(coeff))
  nin,nout = size(x,1),size(y,1)
  n_samples = size(x,2)

  chain = build_lux_chain((nin,red.model.hidden_layers...,nout),red.model.activation)

  bs = resolve_batch_size(red.batch_size,n_samples)
  dataloader = MLUtils.DataLoader((x,y);batchsize=bs,shuffle=true,partial=false)

  Random.seed!(42)
  ps,st = Lux.setup(Random.default_rng(),chain) |> XDEV
  train_state = Lux.Training.TrainState(chain,ps,st,red.optimiser.opt)

  to_device_batch((xb,yb)) = (xb |> XDEV,yb |> XDEV)
  train_model!(
    train_state,dataloader,red.optimiser.lr_scheduler,to_device_batch;logger=red.trainlog
  )
end

# transient 

function train(
  red::DeepONetReduction,
  feop::ODEParamOperator,
  s::AbstractSnapshots
  )

  # Data extraction
  sx = CoordinateSnapshots(s,get_test(feop))
  target = sample(red,sx)
  data,params,coords = get_formatted_data(Float32,target)

  # Normalisation
  stats = NormStats(data,params,coords;normalise=true)

  # Building the DeepONet
  rng = Random.default_rng()
  Random.seed!(rng,42)

  model = build_model(red)
  opt = get_optimiser(red)
  coords_dev = coords |> XDEV
  ps,st = Lux.setup(rng,model) |> XDEV
  train_state = Lux.Training.TrainState(model,ps,st,opt)

  # Dataloader and setup
  bs = resolve_batch_size(red,num_params(s))
  dataloader = MLUtils.DataLoader(
    (params,data);
    batchsize=bs,
    shuffle=true,
    partial=false
  )

  # Executing the pipeline
  trained = train_deeponet!(train_state,dataloader,coords_dev,red)

  return trained,stats
end

function train(
  red::DeepONetReduction,
  feop::ODEParamOperator,
  s::AbstractSnapshots,
  pretrained_op::NeuralOperator;
  update_stats::Bool=false
  )

  # Data extraction
  sx = CoordinateSnapshots(s,get_test(feop))
  target = sample(red,sx)
  data,params,coords = get_formatted_data(Float32,target)

  # Normalisation
  if update_stats
    stats = NormStats(data,params,coords;normalise=true)
  else
    stats = pretrained_op.metadata
    normalise!((data,params,coords),stats)
  end

  # Pretrained model
  model = pretrained_op.model.chain
  opt = get_optimiser(red)
  coords_dev = coords |> XDEV
  ps = pretrained_op.model.parameters |> XDEV
  st = pretrained_op.model.states |> XDEV
  train_state = Lux.Training.TrainState(model,ps,st,opt)

  # Dataloader and setup
  bs = resolve_batch_size(red,num_params(s))
  dataloader = MLUtils.DataLoader(
    (params,data);
    batchsize=bs,
    shuffle=true,
    partial=false
  )

  # Executing the pipeline
  trained = train_deeponet!(train_state,dataloader,coords_dev,red)

  return trained,stats
end

function train(
  red::NOMADReduction,
  feop::ODEParamOperator,
  s::AbstractSnapshots
  )

  # Data extraction
  sx = CoordinateSnapshots(s,get_test(feop))
  target = sample(red,sx)
  data,params,coords = get_formatted_data(Float32,target)
  dout,pin,xin = _flatten(data,params,coords) # Flattening for NOMAD
  N_tot = size(dout,2)

  # Normalisation
  stats = NormStats(dout,pin,xin;normalise=true)

  # Building the NOMAD model
  rng = Random.default_rng()
  Random.seed!(rng,42)

  model = build_model(red)
  opt = get_optimiser(red)
  ps,st = Lux.setup(rng,model) |> XDEV
  train_state = Lux.Training.TrainState(model,ps,st,opt)

  # DataLoader and Lux setup
  bs = resolve_batch_size(red,N_tot)
  dataloader = MLUtils.DataLoader(
    ((pin,xin),dout);
    batchsize=bs,
    shuffle=true,
    partial=false
  )

  # Running the pipeline
  trained = train_nomad!(train_state,dataloader,red)

  return trained,stats
end

function train(
  red::NOMADReduction,
  feop::ODEParamOperator,
  s::AbstractSnapshots,
  pretrained_op::NeuralOperator;
  update_stats::Bool=false
  )

  # Data extraction
  sx = CoordinateSnapshots(s,get_test(feop))
  target = sample(red,sx)
  data,params,coords = get_formatted_data(Float32,target)
  dout,pin,xin = _flatten(data,params,coords) # Flattening for NOMAD
  N_tot = size(dout,2)

  # Normalisation
  if update_stats
    stats = NormStats(dout,pin,xin;normalise=true)
  else
    stats = pretrained_op.metadata
    normalise!((dout,pin,xin),stats)
  end

  # Pretrained model
  model = pretrained_op.model.chain
  opt = get_optimiser(red)
  ps = pretrained_op.model.parameters |> XDEV
  st = pretrained_op.model.states |> XDEV
  train_state = Lux.Training.TrainState(model,ps,st,opt)

  # DataLoader and Lux setup
  bs = resolve_batch_size(red,N_tot)
  dataloader = MLUtils.DataLoader(
    ((pin,xin),dout);
    batchsize=bs,
    shuffle=true,
    partial=false
  )

  # Running the pipeline
  trained = train_nomad!(train_state,dataloader,red)

  return trained,stats
end

# utils

function resolve_batch_size(batch_config::Int,total_samples::Int)
  return batch_config <= 0 ? total_samples : min(batch_config,total_samples)
end

function resolve_batch_size(red::NeuralReduction,total_samples::Int)
  resolve_batch_size(red.batch_size,total_samples)
end

function _flatten(
  data::AbstractArray{T},
  params::AbstractArray{T},
  coords::AbstractArray{T}
  ) where T 

  ntot = size(coords,2)*size(params,2)
  pin = zeros(T,size(params,1),ntot)
  xin = zeros(T,size(coords,1),ntot)
  dout = zeros(T,1,ntot)

  col_idx = 1
  @views for i in axes(params,2)
    p = params[:,i]
    for j in axes(coords,2)
      pin[:,col_idx] = p
      xin[:,col_idx] = coords[:,j]
      dout[1,col_idx] = data[j,i]
      col_idx += 1
    end
  end

  return dout,pin,xin
end

function _flatten(
  params::AbstractArray{T},
  coords::AbstractArray{T}
  ) where T

  ntot = size(coords,2)*size(params,2)
  pin = zeros(T,size(params,1),ntot)
  xin = zeros(T,size(coords,1),ntot)

  col_idx = 1
  @views for i in axes(params,2)
    p = params[:,i]
    for j in axes(coords,2)
      pin[:,col_idx] = p
      xin[:,col_idx] = coords[:,j]
      col_idx += 1
    end
  end

  return pin,xin
end