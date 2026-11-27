# NonlinearROMs

| **Documentation** |
|:------------ |
| [![docdev](https://img.shields.io/badge/docs-dev-blue.svg)](https://nichomueller.github.io/NonlinearROMs.jl/dev/) |
| **Build Status** |
| [![CI](https://github.com/nichomueller/NonlinearROMs.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/nichomueller/NonlinearROMs.jl/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/nichomueller/NonlinearROMs.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/nichomueller/NonlinearROMs.jl) |

## What

NonlinearROMs provides neural-network surrogates for parameterised partial differential equations (PDEs), written in the Julia programming language. It complements the linear reduced order models of [GridapROMs](https://github.com/gridap/GridapROMs.jl) with *neural operators*: networks that learn the map from the problem data (coordinates, coefficients, forcings) to the finite element solution directly from snapshots, without access to the discretised operators.

NonlinearROMs is built on top of [Gridap](https://github.com/gridap/Gridap.jl) and GridapROMs: parameterised problems and their snapshots are defined exactly as in GridapROMs. Networks are written with [Lux](https://github.com/LuxDL/Lux.jl), compiled with [Reactant](https://github.com/EnzymeAD/Reactant.jl) and differentiated with [Enzyme](https://github.com/EnzymeAD/Enzyme.jl). The library currently supports

- kernel neural operators (Kovachki et al.), with the integral kernel discretised by finite element quadrature on the mesh, or by message passing on a radius graph (graph neural operators)
- Fourier neural operators, with a truncated spectral kernel
- steady and transient, single- and multi-field problems, on fixed or varying meshes
- autoregressive training for transient problems, with temporal bundling and the pushforward trick (Brandstetter et al.)
- training losses in the energy norms of the finite element space (`ℓ²`, `L²`, `H¹`)
- configurable normalisation of the coordinates, features and outputs

A neural operator surrogate of a parameterised Poisson problem takes a few lines of code:

```julia
using Gridap, GridapROMs, NonlinearROMs

pspace = ParamSpace((1,10,1,10))
model = CartesianDiscreteModel((0,1,0,1),(20,20))
Ω = Triangulation(model)
dΩ = Measure(Ω,2)
V = TestFESpace(Ω,ReferenceFE(lagrangian,Float64,1);dirichlet_tags="boundary")
U = ParamTrialFESpace(V)

κ(μ) = x -> 1 + μ[1]*exp(-μ[2]*x[1]*x[2])
κμ(μ) = parameterise(κ,μ)
a(μ,u,v) = ∫(κμ(μ)*∇(v)⋅∇(u))dΩ
l(μ,v) = ∫(v)dΩ
feop = LinearParamFEOperator(l,a,pspace,U,V)
fesolver = LinearFESolver(LUSolver())

μ_train = realisation(pspace;nparams=50)
uh_train = solve(fesolver,feop,μ_train)             # FE snapshots

x = get_coords(V)
inputs(μ) = InputSpace(x,FeatureBundle((κμ(μ),)))   # coordinates and coefficient
fspace = FunctionSpace(V)
nmodel = KernelNeuralModel(fspace;hidden_layers=(64,64))
solver = NeuralSolver(nmodel;epochs=1000,loss=EnergyLoss(H1(),fspace))
op = neural_operator(solver,inputs(μ_train),OutputSpace(uh_train))  # offline phase

μ = realisation(pspace;nparams=10)
û = solve(solver,op,inputs(μ))                      # online phase
```

## Documentation

- [**DEVEL**](https://nicholasmueller.co/NonlinearROMs.jl/dev/) — *Documentation for the most recent in-development version of NonlinearROMs.*

## Tutorials

A hands-on introduction to the library is available as a set of tutorials in the documentation:

1. [Poisson equation](https://nicholasmueller.co/NonlinearROMs.jl/dev/example/)
2. [Stokes flow](https://nicholasmueller.co/NonlinearROMs.jl/dev/stokes/)
3. [Heat equation (rollout)](https://nicholasmueller.co/NonlinearROMs.jl/dev/heat_equation/)
4. [Learning a Green function](https://nicholasmueller.co/NonlinearROMs.jl/dev/green/)

Preliminary results are collected in [Preliminary results: Poisson equation](https://nicholasmueller.co/NonlinearROMs.jl/dev/poisson_results/).

## Installation

NonlinearROMs is not registered yet. It requires Julia 1.11 or later, and can be installed from the Julia REPL with

```julia
using Pkg
Pkg.add(url="https://github.com/nichomueller/NonlinearROMs.jl")
```

## Testing

```julia
using Pkg
Pkg.test("NonlinearROMs")
```
