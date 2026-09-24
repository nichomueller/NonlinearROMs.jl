# NonlinearROMs.jl

Neural-network-based hyper-reduction and nonlinear reduced-order modelling
components for [GridapROMs.jl](https://github.com/gridap/GridapROMs.jl).

This package was extracted from `GridapROMs.RBSteady`/`GridapROMs.RBTransient`
(the `NonlinearModels.jl`, `NNHyperReduction`, `NNOperatorReduction`, and
`HighDimNN*` transient counterparts) into its own repository, and plugs back
into `GridapROMs` via multiple dispatch — no changes to `GridapROMs` itself
are required.

## Installation

`NonlinearROMs` is not a registered package, so it must be added as a `dev` dependency (or otherwise made resolvable) before instantiating this environment:

```julia
using Pkg
Pkg.develop(path="../NonlinearROMs.jl")
Pkg.instantiate()
```

## Usage

```julia
using GridapROMs
using NonlinearROMs

res_reduction = NNHyperReduction(tol; nparams, compression)
jac_reduction = NNOperatorReduction(tol; nparams)
rbsolver = RBSolver(fesolver, state_reduction, res_reduction, jac_reduction)
```

`NNHyperReduction`/`NNOperatorReduction` (and their transient counterparts
`HighDimNNHyperReduction`/`HighDimNNOperatorReduction`) can be passed anywhere
a `HyperReduction` is expected, exactly like `MDEIMHyperReduction` or
`RBFHyperReduction`.

## Contents

- `AbstractNeuralModels.jl` — `MultiLayerPerceptron`,
  `AutoEncoder`, `VariationalAutoEncoder`, `AutoDecoder`, `train_neural_coefficient`,
  `NNStrategy`.
- `SteadyReductions.jl` / `SteadyHyperReductions.jl` / `SteadyInterpolations.jl` /
  `SteadyReducedOperators.jl` — steady `NNOperatorReduction`/`NNHyperReduction`.
- `TransientReductions.jl` / `TransientHyperReductions.jl` /
  `TransientInterpolations.jl` / `TransientReducedOperators.jl` — transient
  `HighDimNNOperatorReduction`/`HighDimNNHyperReduction`.

## Testing

```julia
using Pkg
Pkg.test("NonlinearROMs")
```
