abstract type LRScheduler end

# Helpers

function get_lr(s::LRScheduler)
  @abstractmethod
end

function step_scheduler!(s::LRScheduler,args...;kwargs...)
  @abstractmethod
end

"""
    struct CosineAnnealing <: LRScheduler
      total_epochs::Int
      lr_max::Float32
      lr_min::Float32
    end

A learning rate scheduler that implements a Cosine Annealing decay schedule.
It smoothly decreases the learning rate from a maximum value down to a minimum value, following the shape of a half-cosine wave over `total_epochs` epochs.

# Fields / Keyword Arguments
- `total_epochs::Int`: The total number of training epochs the schedule decays over (required positional argument).
- `lr_max::Float32`: The initial, peak learning rate (default: `0.001f0`).
- `lr_min::Float32`: The final, minimum learning rate at the end of training (default: `1e-6f0`).
"""
struct CosineAnnealing <: LRScheduler
  total_epochs::Int
  lr_max::Float32
  lr_min::Float32
end

CosineAnnealing(total_epochs;lr_max=0.001f0,lr_min=1f-6) = CosineAnnealing(total_epochs,lr_max,lr_min)

get_lr(s::CosineAnnealing) = s.lr_max

function step_scheduler!(s::CosineAnnealing,opt_state,epoch::Int,args...;kwargs...)
  t = min(epoch,s.total_epochs)
  cos_val = cos(π * (t / s.total_epochs))
  new_lr = s.lr_min + 0.5f0 * (s.lr_max - s.lr_min) * (1.0f0 + Float32(cos_val))
  Optimisers.adjust!(opt_state,new_lr)
end

"""
    mutable struct ReduceLROnPlateau <: LRScheduler
      patience::Int
      factor::Float32
      min_lr::Float32
      wait::Int
      best_loss::Float32
      current_lr::Float32
    end

A dynamic learning rate scheduler that reduces the learning rate by a multiplicative `factor` when the training loss has stopped improving for a specified number of epochs (`patience`).

# Keyword Arguments
- `patience::Int`: Number of epochs to wait without loss improvement before reducing the learning rate (default: `100`).
- `factor::Float32`: The multiplicative factor applied to the learning rate upon plateauing (default: `0.5f0`).
- `min_lr::Float32`: The absolute minimum learning rate boundary. The scheduler will not decay below this value (default: `1e-6f0`).
- `start_lr::Float32`: The initial learning rate at the beginning of the training (default: `0.001f0`).
"""
struct ReduceLROnPlateau <: LRScheduler
  patience::Int
  factor::Float32
  min_lr::Float32
  wait::Base.Ref{Int}
  best_loss::Base.Ref{Float32}
  current_lr::Base.Ref{Float32}
end

function ReduceLROnPlateau(;patience=100,factor=0.5f0,min_lr=1f-6,start_lr=0.001f0)
  ReduceLROnPlateau(patience,factor,min_lr,Base.Ref(0),Base.Ref(Inf32),Base.Ref(start_lr))
end

get_lr(s::ReduceLROnPlateau) = s.current_lr[]

function step_scheduler!(s::ReduceLROnPlateau,opt_state,epoch,current_loss;verbose::Bool=false)
  if current_loss < s.best_loss[]
    s.best_loss[] = current_loss
    s.wait[] = 0
  else
    s.wait[] += 1
  end

  if s.wait[] >= s.patience
    new_lr = max(s.current_lr[] * s.factor,s.min_lr)
    if new_lr < s.current_lr[]
      verbose && @info "Plateau reached: LR decreased from $(s.current_lr[]) to $new_lr"
      s.current_lr[] = new_lr
      Optimisers.adjust!(opt_state,new_lr)
    end
    s.wait[] = 0
  end
end

struct Optimiser
  opt::Optimisers.AbstractRule 
  lr_scheduler::LRScheduler
end

function Optimiser(;
  lr_scheduler::LRScheduler,
  opt=Optimisers.Adam(get_lr(lr_scheduler)),
  weight_decay::Real=0.0
  )

  opt = weight_decay > 0 ? Optimisers.OptimiserChain(opt,Optimisers.WeightDecay(weight_decay)) : opt
  Optimiser(opt,lr_scheduler)
end