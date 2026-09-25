# NonlinearROMs.jl Release Notes

## Review of the Neural Operators PR

This branch is a review of Isaia's neural-operators PR (merged at `d6f6c18`, "merged, starting review"). The goal of this note is to separate what changed (and why) from what didn't. Most of the *algorithmic* code Isaia wrote — the Lux/Reactant/Enzyme training loop, the device-transfer pattern, the DeepONet/NOMAD architectures themselves, the GridapROMs integration points — is untouched. The changes that have been made aim to:
- Simplify the code by finding common structures/operations used in different parts of the library, and replacing the old code by use these new functions instead. This will improve extendibility for future releases.
- Integrate Isaia's architecture with my pre-existing neural networks. The result is that we have already developed a fairly mature suite of nonlinear models in our package. 

### What stayed the same

- **The training loop itself.** `train_model!`'s core — `Lux.Training.single_train_step!` with `Lux.AutoEnzyme()`/`Lux.MSELoss()`, wrapped in
  `Reactant.with_config(;dot_general_precision=Reactant.PrecisionConfig.HIGH)` — is exactly as Isaia wrote it.
- **Device transfer.** `CDEV = Lux.cpu_device()` / `XDEV = Lux.reactant_device(;force=true)` and where each is applied (device before training, CPU after) is unchanged.
- **The architectures.** `LuxDeepONet`/`LuxNOMAD` (the `Lux.Parallel`-based branch/trunk and approximator/decoder chains) are the same networks.
- **The scheduler math.** Cosine annealing and plateau-reduction compute the same learning rate schedules as before; only how they're stored/invoked changed (below).
- **The overall GridapROMs integration shape.** `RBSolver` / `reduced_operator` / `Algebra.solve` multiple-dispatch pattern is the same one Isaia set up; this package still plugs into `RBSteady`/`RBTransient` the same way.

### What changed, and why

**1. File reorganization: split along the seams the code was actually growing along.**
I changed several file names and introduced new files, so that the overall structure of the package is the following:

  | File | Now holds |
  |---|---|
  | `AbstractNeuralModels.jl` | the `AbstractNeuralModel` abstract type only |
  | `AbstractNeuralModels.jl` | DeepONet/NOMAD/MLP/AutoEncoder/VAE/AutoDecoder architectures |
  | `AbstractNeuralModelsTraining.jl` | `TrainedAbstractNeuralModel`, `train_model!`, generic trained-network wrappers |
  | `NeuralReductions.jl` | `NeuralReduction`/`DeepONetReduction`/`NOMADReduction` |
  | `NeuralSolvers.jl` / `TransientNeuralSolver.jl` | `NeuralOperator`, `reduced_operator`, `Algebra.solve` |
  | `NeuralTraining.jl` / `TransientNeuralTraining.jl` | the DeepONet/NOMAD `train(...)` pipelines |
  | `Samplers.jl` | `Sampler`/`MultiSampler` |
  | `NeuralLayers.jl` | `LatentCodeLayer`/`VAELayer` |
  | `Utils.jl` | absorbed `TrainingLogs.jl`, plus `ZScore`/`Normalisation`/`normalise!`/`CoordinateSnapshots`/`get_formatted_data` |

This is purely organizational: no behavior changed, other than the load-order constraints Julia imposes when a type moves to a file that now `include`s too late for another file's function signature to reference it (hit and fixed a few times over the course of this review).

**2. One `NeuralReduction` for every neural-network use case, not two.**
Before, DeepONet/NOMAD had their own training configuration, while the MLP-based hyper-reduction regressors (`NNRegression`, `NNHyperReduction`) had a completely separate `NNStrategy` (`type=MLPType()`, `layers`, `lr`, `optimiser`, `loss`, `epochs`, `weight_decay`, `batch_size`, `lr_schedule`, `patience`, `val_fraction`). Now a single
`NeuralReduction{A<:AbstractNeuralModel}` (model + epochs + batch_size + sampler + optimiser + training log) configures all of them. Also, I removed the `AutoDeepONet`/`AutoNOMAD` structs, as these can easily be merged with the non-`Auto` versions. Consequently, `resolve_model` was also removed.

**3. One subsampling abstraction (`Sampler`/`MultiSampler`) instead of separate steady/transient code.**
Selecting a subset of spatial DoFs, parameters, and (for transient problems) time steps before training used to be separate, independently-written logic for the steady and transient cases. It's now one `sample(...)` multiple-dispatch entry point, driven by a
`MultiSampler(space_sampler,param_sampler,time_sampler)` where each sub-sampler is an `Integer` stride, a `Function` transform, an explicit index vector, or `nothing`/`identity` (no subsampling). This allows us to cleanly select the datasets for training and testing without having to rewrite every time several (cumbersome) `for` loops.

**4. Learning-rate schedulers: encapsulated instead of re-passed every call.**
`AbstractLRScheduler` → `LRScheduler`; `get_initial_lr` → `get_lr`. `CosineAnnealing` now stores `total_epochs` as a field set once at construction, instead of requiring every `step_scheduler!` call to pass it in. `ReduceLROnPlateau`'s mutable counters (`wait`/`best_loss`/`current_lr`) became `Base.Ref`s on an immutable `struct` rather than
fields of a `mutable struct`. The cosine/plateau math itself is unchanged.

**5. `NeuralOperator`: weights and states now live inside the trained model.**
Before, `NeuralOperator` carried `model` (a bare Lux chain), `model_weights`, `model_states`, `norm_stats` and `max_u` as separate fields. I see these fields as two macro-structures: 
- Chain + parameters + states: these make up a single `TrainedAbstractNeuralModel`.
- The rest can be seen as a structure collecting information on the normalisation factors of the data. By default, this structure is a `Normalisation` which basically contains the old `norm_stats` and `max_u`; however, this could also be of type `Nothing`, if no normalisation is applied.
In essence, now a `NeuralOperator` contains only two fields: a `model <: TrainedAbstractNeuralModel`, and `metadata <: Union{Normalisation, Nothing}`.

**6. Now using my proposed coordinates builder.**

**7. `AutoEncoder`/`AutoDecoder`/`VariationalAutoEncoder` now go through `reduced_operator` too.**
