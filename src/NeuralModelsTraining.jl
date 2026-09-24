const CDEV = Lux.cpu_device()
const XDEV = Lux.reactant_device(;force=true)

"""
    struct TrainedAbstractNeuralModel{A,B,C} <: AbstractNeuralModel
      chain::A
      parameters::B
      states::C
    end

A trained Lux `chain` bundled with its optimised parameters/states, evaluable
as `(a::TrainedAbstractNeuralModel)(x::AbstractMatrix) -> AbstractMatrix` via the standard
`Arrays.evaluate!`/`return_cache` interface. Returned by [`train_neural_coefficient`](@ref)
for [`MultiLayerPerceptron`](@ref) strategies.
"""
struct TrainedAbstractNeuralModel{A,B,C} <: AbstractNeuralModel
  chain::A
  parameters::B
  states::C
end

function TrainedAbstractNeuralModel(train_state::Lux.Training.TrainState)
  parameters = train_state.parameters |> CDEV
  states = Lux.testmode(train_state.states) |> CDEV
  TrainedAbstractNeuralModel(train_state.model,parameters,states)
end

function Arrays.evaluate!(cache,a::TrainedAbstractNeuralModel,x::AbstractMatrix)
  first(a.chain(Float32.(x),a.parameters,a.states))
end

"""
    (m::TrainedAbstractNeuralModel)(inputs) -> AbstractArray
    (m::TrainedAbstractNeuralModel)(inputs,metadata) -> AbstractArray

Applies `m` to `inputs` (a `(params,coords)`/`(pin,xin)` tuple for DeepONet/NOMAD, or
a plain matrix). `metadata` optionally denormalises the output by `metadata.dmax`; it
is a no-op when `metadata === identity` (a `NeuralOperator` with no normalisation stats).
"""
function (m::TrainedAbstractNeuralModel)(inputs)
  first(m.chain(inputs,m.parameters,m.states))
end

(m::TrainedAbstractNeuralModel)(inputs,metadata::typeof(identity)) = m(inputs)

function (m::TrainedAbstractNeuralModel)(inputs,metadata::NormStats)
  pred = m(inputs)
  pred .*= metadata.dmax
  return pred
end

const TrainedAutoEncoder = TrainedAbstractNeuralModel{<:AutoEncoder}

function Arrays.evaluate!(cache,a::TrainedAutoEncoder,z::AbstractMatrix)
  decode(a,z)
end

function encode(a::TrainedAutoEncoder,X::AbstractMatrix)
  first(a.chain.layers.layer_1(Float32.(X),a.parameters.layer_1,a.states.layer_1))
end

function decode(a::TrainedAutoEncoder,Z::AbstractMatrix)
  first(a.chain.layers.layer_2(Float32.(Z),a.parameters.layer_2,a.states.layer_2))
end

const TrainedAutoDecoder = TrainedAbstractNeuralModel{<:AutoDecoder}

function Arrays.evaluate!(cache,a::TrainedAutoDecoder,z::AbstractMatrix)
  first(a.chain.layers.layer_2(Float32.(z),a.parameters.layer_2,a.states.layer_2))
end

get_latent_codes(a::TrainedAutoDecoder) = a.parameters.layer_1.codes

"""
    infer_latent(a::TrainedAutoDecoder,x_target::AbstractVector,r::NeuralReduction) -> AbstractVector

Fit a latent code `z` for an unseen snapshot `x_target` by minimising the mean
squared reconstruction error with the decoder weights fixed, using `r.optimiser.opt`
and `r.epochs`.
"""
function infer_latent(a::TrainedAutoDecoder,x_target::AbstractVector,r::NeuralReduction)
  latent_dim = size(get_latent_codes(a),1)
  T = Float32
  z = randn(T,latent_dim) .* T(0.01)
  X_t = reshape(T.(x_target),:,1)
  opt_state = Optimisers.setup(r.optimiser.opt,z)
  for _ in 1:r.epochs
    grad = ForwardDiff.gradient(z) do z_
      X̂ = first(a.chain.layers.layer_2(reshape(z_,:,1),a.parameters.layer_2,a.states.layer_2))
      sum(abs2,X̂ .- X_t)/length(X_t)
    end
    opt_state,z = Optimisers.update!(opt_state,z,grad)
  end
  z
end

"""
    struct TrainedVAE{E,D,PE,SE,PD,SD} <: AbstractNeuralModel
      encoder::E
      decoder::D
      ps_enc::PE
      st_enc::SE
      ps_dec::PD
      st_dec::SD
      latent_dim::Int
    end

A trained [`VariationalAutoEncoder`](@ref). `evaluate!(cache,a,z)` applies the
**decoder** (latent → high-dim); use [`encode`](@ref) for the encoder direction,
which returns `(μ,log_var,z)` with a freshly sampled `z`.
"""
struct TrainedVAE{E,D,PE,SE,PD,SD} <: AbstractNeuralModel
  encoder::E
  decoder::D
  ps_enc::PE
  st_enc::SE
  ps_dec::PD
  st_dec::SD
  latent_dim::Int
end

function Arrays.evaluate!(cache,a::TrainedVAE,z::AbstractMatrix)
  decode(a,z)
end

function encode(a::TrainedVAE,X::AbstractMatrix)
  enc_out = first(a.encoder(Float32.(X),a.ps_enc,a.st_enc))
  μ = @views enc_out[1:a.latent_dim,:]
  log_var = @views enc_out[a.latent_dim+1:end,:]
  ε = randn(eltype(μ),size(μ))
  z = μ .+ ε .* exp.(log_var ./ 2)
  (μ,log_var,z)
end

function decode(a::TrainedVAE,Z::AbstractMatrix)
  first(a.decoder(Float32.(Z),a.ps_dec,a.st_dec))
end

"""
    train_model!(train_state,dataloader,lr_scheduler,to_device_batch;loss=Lux.MSELoss(),logger::TrainingLog)

Generic Lux/Reactant/Enzyme training loop shared by every architecture in this package.
`to_device_batch` maps one raw batch yielded by `dataloader` to whatever `loss` expects
as its data argument (already moved to `XDEV`) — DeepONet pairs each batch with a fixed
set of trunk query points, NOMAD's coordinates are already part of the per-row batch,
and a plain reconstruction network (AutoEncoder/AutoDecoder) just needs `(x,x)`. `loss`
defaults to `Lux.MSELoss()`; pass a custom `(model,ps,st,data) -> (loss,st,stats)`
function for anything else (e.g. a VAE's reconstruction+KL loss).
"""
function train_model!(train_state,dataloader,lr_scheduler,to_device_batch;loss=Lux.MSELoss(),logger::TrainingLog)
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
        current_loss += Float32(loss_val)
      end
      current_loss /= length(dataloader)

      step_scheduler!(lr_scheduler,train_state.optimizer_state,epoch,current_loss;verbose=logger.verbose)

      update!(logger,epoch,current_loss)
    end
  end

  finalize!(logger)
  return TrainedAbstractNeuralModel(train_state)
end