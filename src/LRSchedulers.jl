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
      lr_max::TData
      lr_min::TData
    end

A learning rate scheduler that implements a Cosine Annealing decay schedule.
It smoothly decreases the learning rate from a maximum value down to a minimum value, following the shape of a half-cosine wave over `total_epochs` epochs.

# Fields / Keyword Arguments
- `total_epochs::Int`: The total number of training epochs the schedule decays over (required positional argument).
- `lr_max::TData`: The initial, peak learning rate (default: `TData(0.001)`).
- `lr_min::TData`: The final, minimum learning rate at the end of training (default: `TData(1e-6)`).
"""
struct CosineAnnealing <: LRScheduler
  total_epochs::Int
  lr_max::TData
  lr_min::TData
end

CosineAnnealing(total_epochs;lr_max=TData(0.001),lr_min=TData(1e-6)) = CosineAnnealing(total_epochs,lr_max,lr_min)

get_lr(s::CosineAnnealing) = s.lr_max

function step_scheduler!(s::CosineAnnealing,opt_state,epoch::Int,args...;kwargs...)
  t = min(epoch,s.total_epochs)
  cos_val = cos(π * (t / s.total_epochs))
  new_lr = s.lr_min + TData(0.5) * (s.lr_max - s.lr_min) * (TData(1.0) + TData(cos_val))
  Optimisers.adjust!(opt_state,new_lr)
end

"""
    mutable struct ReduceLROnPlateau <: LRScheduler
      patience::Int
      factor::TData
      min_lr::TData
      wait::Int
      best_loss::TData
      current_lr::TData
    end

A dynamic learning rate scheduler that reduces the learning rate by a multiplicative `factor` when the training loss has stopped improving for a specified number of epochs (`patience`).

# Keyword Arguments
- `patience::Int`: Number of epochs to wait without loss improvement before reducing the learning rate (default: `100`).
- `factor::TData`: The multiplicative factor applied to the learning rate upon plateauing (default: `TData(0.5)`).
- `min_lr::TData`: The absolute minimum learning rate boundary. The scheduler will not decay below this value (default: `TData(1e-6)`).
- `start_lr::TData`: The initial learning rate at the beginning of the training (default: `TData(0.001)`).
"""
mutable struct ReduceLROnPlateau <: LRScheduler
  const patience::Int
  const factor::TData
  const min_lr::TData
  wait::Int
  best_loss::TData
  current_lr::TData
end

function ReduceLROnPlateau(;
  patience=100,
  factor=TData(0.5),
  min_lr=TData(1e-6),
  start_lr=TData(0.001)
  )

  ReduceLROnPlateau(patience,factor,min_lr,0,TData(Inf),start_lr)
end

get_lr(s::ReduceLROnPlateau) = s.current_lr

function step_scheduler!(
  s::ReduceLROnPlateau,
  opt_state,
  epoch,
  current_loss;
  verbose::Bool=false
  )

  if current_loss < s.best_loss
    s.best_loss = current_loss
    s.wait = 0
  else
    s.wait += 1
  end

  if s.wait >= s.patience
    new_lr = max(s.current_lr * s.factor,s.min_lr)
    if new_lr < s.current_lr
      verbose && @info "Plateau reached: LR decreased from $(s.current_lr) to $new_lr"
      s.current_lr = new_lr
      Optimisers.adjust!(opt_state,new_lr)
    end
    s.wait = 0
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

  opt = if weight_decay > 0
    Optimisers.OptimiserChain(opt,Optimisers.WeightDecay(weight_decay)) 
  else
    opt
  end
  Optimiser(opt,lr_scheduler)
end