# Neural Operators

`NonlinearROMs.jl` builds non-intrusive surrogates of parametric FE problems as
*neural operators*: networks mapping input functions (coordinates and PDE
coefficients) to output functions (FE solutions), trained with
[Lux.jl](https://github.com/LuxDL/Lux.jl) on an XLA device via
[Reactant.jl](https://github.com/EnzymeAD/Reactant.jl), with gradients computed
by [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl).

## Data

Every per-point quantity is stored as a dense `(D,N,K)` array (`D` channels,
`N` points, `K` samples).

- An [`InputSpace`](@ref) evaluates the coordinates and a [`FeatureBundle`](@ref)
  of `ParamFunction`s on a point set.
- An [`OutputSpace`](@ref) evaluates the FE solution on the same points.
- [`to_element`](@ref) turns a prediction back into an `FEFunction`.

Multi-field and multi-mesh problems are handled by combining several such
spaces (see the [Stokes](stokes.md) and [moving Poisson](moving_poisson.md)
tutorials).

## Models

A [`KernelNeuralModel`](@ref) is the iterative kernel-integral architecture of
Kovachki et al.:

```math
v_{t+1}(x) = \sigma\big(W v_t(x) + b + (K_t v_t)(x)\big).
```

It consists of a pointwise lift, one [`KernelNeuralLayer`](@ref) per interior
hidden width, and a pointwise projection. Its `style` decides how the
non-local operator ``K_t`` is discretised:

- an `IntegralStyle`, built from a discrete space, computes
  ``\int \kappa_\theta(x,y,\ldots)\,v(y)\,\mathrm dy`` against an
  [`InducedMeasure`](@ref):
  - [`FunctionSpace`](@ref) uses FE quadrature (mass-matrix weights).
  - [`GraphInducedSpace`](@ref) and [`MultiGraphInducedSpace`](@ref) use
    message passing on a radius graph (GNO).
- [`FourierStyle`](@ref) uses a spectral convolution with a truncated number of
  Fourier modes (FNO, see [`FourierKernel`](@ref)).

A [`DataFormatter`](@ref) selects which of the coordinates, the features and the
running hidden state are fed to the kernel network ``\kappa_\theta``.

## Normalisation

Inputs and outputs are normalised before training, and predictions are
denormalised after evaluation. The recipe is a `NormStyle`, passed to the
[`DataFormatter`](@ref) as `norm_style`, which picks one score style for each of
the coordinates, the features and the outputs:

- [`MScoreStyle`](@ref): division by the maximum absolute value;
- [`RScoreStyle`](@ref): an affine map to ``[0,1]``;
- [`ZScoreStyle`](@ref): zero mean and unit variance per channel.

The resulting statistics are stored in the trained [`NeuralOperator`](@ref).

## Training

[`NeuralSolver`](@ref) bundles the model with the loss, the optimiser, the
learning-rate schedule and the number of epochs. The default loss is the MSE.
[`EnergyLoss`](@ref) instead measures the error in one of `GridapROMs`' norms
(`ℓ2()`, `L2()`, `H1()`, or a `BlockNorm` for multi-field problems).

- [`neural_operator`](@ref) runs the offline phase: it fetches and normalises
  the data, then trains the network. It returns a [`NeuralOperator`](@ref).
- `solve(solver,op,input_space)` runs the online phase.
- [`replace_space`](@ref) evaluates a trained kernel operator on a new point
  set or mesh.

## Transient problems: rollout

Transient problems are learned autoregressively, with the temporal bundling
and pushforward trick of Brandstetter et al.

- [`RolloutData`](@ref) cuts the time steps into clusters, each split into two
  stages.
- Evaluating it in an [`InputSpace`](@ref)/[`OutputSpace`](@ref) gives
  [`TwoStageBuckets`](@ref).
- A [`RolloutNeuralSolver`](@ref) trains the network to advance the state from
  one stage to the next, with a stop-gradient pushforward term that makes it
  robust to its own errors.
- `solve(solver,op,input_space,initial_output_space)` rolls the network out
  from the initial condition (see [`rollout`](@ref)).

The [heat equation tutorial](heat_equation.md) works through an example.
