using Documenter
using Literate
using NonlinearROMs

Literate.markdown(
  joinpath(@__DIR__,"literate","poisson.jl"),
  joinpath(@__DIR__,"src");
  name="example",
  documenter=true,
)

Literate.markdown(
  joinpath(@__DIR__,"literate","stokes.jl"),
  joinpath(@__DIR__,"src");
  name="stokes",
  documenter=true,
)

Literate.markdown(
  joinpath(@__DIR__,"literate","heat_equation.jl"),
  joinpath(@__DIR__,"src");
  name="heat_equation",
  documenter=true,
)

# `green.jl` trains for several minutes and plots with Plots.jl (not a docs
# dependency): its blocks are rendered as plain Julia code, illustrated by the
# figures it produced, stored in `src/assets/green`.
Literate.markdown(
  joinpath(@__DIR__,"literate","green.jl"),
  joinpath(@__DIR__,"src");
  name="green",
  documenter=true,
  codefence="```julia" => "```",
)

# `moving_poisson.jl` is a documented work-in-progress (see the warning at the
# top of the generated page): its code does not run yet, so its blocks are
# rendered as plain (non-executed) Julia code rather than `@example` blocks.
Literate.markdown(
  joinpath(@__DIR__,"literate","moving_poisson.jl"),
  joinpath(@__DIR__,"src");
  name="moving_poisson",
  documenter=true,
  codefence="```julia" => "```",
)

makedocs(;
  modules=[NonlinearROMs],
  format=Documenter.HTML(size_threshold=nothing),
  pages=[
    "Home" => "index.md",
    "Description" => [
      "Neural Operators" => "neurals.md",
    ],
    "Tutorials" => [
      "Poisson equation" => "example.md",
      "Stokes flow" => "stokes.md",
      "Heat equation (rollout)" => "heat_equation.md",
      "Learning a Green function" => "green.md",
      "Moving Poisson (WIP)" => "moving_poisson.md",
    ],
    "Results" => [
      "Preliminary results: Poisson equation" => "poisson_results.md",
    ],
  ],
  sitename="NonlinearROMs.jl",
  warnonly=[:cross_references,:missing_docs],
)

deploydocs(
  repo="github.com/nichomueller/NonlinearROMs.jl.git",
  push_preview=true,
)
