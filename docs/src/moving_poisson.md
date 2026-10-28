```@meta
EditURL = "../literate/moving_poisson.jl"
```

# Tutorial: towards a Neural Operator on a *moving* geometry (work in progress)

!!! warning "Work in progress"
    This tutorial documents [`test/moving_poisson.jl`](https://github.com/nichomueller/NonlinearROMs.jl/blob/main/test/moving_poisson.jl)
    as it stands today: **it does not run yet**. Several variables used
    below (`loss`, `features`, `dir`, `μ_train`, `output_space`, `x`) are
    never defined in the source script, and the file ends mid-exploration
    of Gridap's dof/node machinery. It is included here so the intended
    shape of the example is documented, not as a working recipe to copy.

This is a short companion to the
[Poisson tutorial](@ref "Tutorial: a Neural Operator surrogate for a parametric Poisson problem")
and the [Stokes tutorial](@ref "Tutorial: a Neural Operator surrogate for parametric Stokes flow"),
which it assumes you have read. The new idea here: instead of one fixed
mesh, the PDE is solved on an **embedded, parameter-dependent geometry** --
a disk of radius ``R`` cut out of a fixed background mesh, both ``R`` and
the disk's centre ``x_0`` varying across samples -- using `GridapEmbedded`.
Every sample can therefore live on a genuinely different active mesh, which
is exactly the "case 3" of the file-top comment in `DiscreteSpaces.jl`
(several spaces, several meshes, no shared point set to fall back on).

```julia
using DrWatson
using Gridap
using GridapEmbedded
using GridapROMs
using NonlinearROMs

const L = 2
const W = 2
const n = 20
const γd = 10.0
const hd = max(L,W)/n

domain = (0,L,0,W)
partition = (n,n)
bgmodel = CartesianDiscreteModel(domain,partition)

order = 1
degree = 2*order

pdomain = (1,10,1,10,0,1)
pspace = ParamSpace(pdomain)

reffe = ReferenceFE(lagrangian,Float64,order)

a(μ) = x -> μ[1]
aμ(μ) = parameterise(a,μ)

f(μ) = x -> μ[2]
fμ(μ) = parameterise(f,μ)

g(μ) = x -> μ[3]
gμ(μ) = parameterise(g,μ)
```

## 1. A geometry-as-a-function-of-parameters helper

`pde(R,x0)` is *not* parameterised over `μ` the way `a`/`f`/`g` are -- `R`
and `x0` are plain numbers fixed once per call, each producing its own
`AgFEMSpace` (an aggregated, cut-cell-stabilised FE space from
`GridapEmbedded`) on the disk-shaped domain they cut out of `bgmodel`. This
is the key structural difference from the Poisson/Stokes tutorials: there
the mesh was one fixed object reused for every `μ`; here, the mesh itself
is swapped out per call, and `μ` only still varies the PDE's coefficients
*within* a given mesh.

```julia
function pde(R,x0)
  geo = !disk(R,x0=x0)
  cutgeo = cut(bgmodel,geo)

  Ωact = Triangulation(cutgeo,ACTIVE)
  Ω = Triangulation(cutgeo,PHYSICAL)
  Γ = EmbeddedBoundary(cutgeo)

  dΩ = Measure(Ω,degree)
  dΓ = Measure(Γ,degree)

  n_Γ = get_normal_vector(Γ)
  strategy = AggregateAllCutCells()
  aggregates = aggregate(strategy,cutgeo)

  stiffness(μ,u,v) = ∫(aμ(μ)*∇(v)⋅∇(u))dΩ + ∫( (γd/hd)*v*u - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ
  rhs(μ,v) = ∫(fμ(μ)*v)dΩ

  testact = TestFESpace(Ωact,reffe,conformity=:H1,dirichlet_tags=[1,3,7])
  test = AgFEMSpace(testact,aggregates)
  trial = ParamTrialFESpace(test,gμ)
  feop = LinearParamFEOperator(rhs,stiffness,pspace,trial,test)

  return (test,trial,feop)
end

fesolver = LinearFESolver(LUSolver())
```

## 2. Three different disks, three different meshes

`test1`/`test2` are two training geometries (and, below, would need their
own disjoint `realisation`/`solve` pairs rather than the shared
`μ_train`/`uhμ_train` names the current draft reuses); `test`/`uhμ` is a
third, held-out geometry for evaluation.

```julia
test1,trial1,feop1 = pde(0.1,Point(0.5,1.5))
μ_train1 = realisation(pspace;nparams=25)
uhμ_train1 = solve(fesolver,feop1,μ_train1)

test2,trial2,feop2 = pde(0.2,Point(1.5,0.5))
μ_train2 = realisation(pspace;nparams=25)
uhμ_train2 = solve(fesolver,feop2,μ_train2)

test,trial,feop = pde(0.3,Point(1.0,1.0))
μ = realisation(pspace;nparams=1)
uhμ = solve(fesolver,feop,μ)
```

## 3. Why this needs `MultiGraphInducedSpace`, not `GraphInducedSpace`

`GraphInducedSpace(test;radius=...)` (used by the Poisson/Stokes tutorials)
builds one radius graph on one point set. That is not meaningful across
`test1`/`test2`'s two *different* meshes -- there is no single shared point
set to build a mass matrix or one graph over (unlike Stokes' velocity/
pressure pair, whose points at least share one triangulation).
[`MultiGraphInducedSpace`](@ref) instead builds one `GraphInducedSpace`
*per* input space and reconciles them onto the union of all their points,
analogous to how [`join_spaces`](@ref) reconciles `OutputSpace`s for a
`MultiFieldFESpace` -- the general mechanism for "several spaces, several
meshes" alluded to in the Stokes tutorial's §2.

The remaining lines sketch the intended training call (mirroring the
Poisson/Stokes tutorials' §8-9), but as flagged above, `loss`, `features`,
`dir`, `μ_train`, `output_space` and `x` are not yet defined anywhere in
the source script -- this is as far as the current draft goes.

```julia
gspace = MultiGraphInducedSpace(test1,test2;radius=2*hd)
gmodel = KernelNeuralModel(gspace;hidden_layers=(16,16),activation=tanh)
solver = NeuralSolver(fesolver,gmodel;epochs=200,loss=loss,print_every=50)
input_space = InputSpace((x,features(μ)))

op = neural_operator(solver,input_space(μ_train),output_space)
pred_uμ = solve(solver,op,input_space(μ))
pred_uhμ = to_element(OutputSpace(uhμ),pred_uμ)
visualise(dir,uhμ,pred_uhμ;variable_name="uh_gno")
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

