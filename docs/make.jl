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
      "Moving Poisson (WIP)" => "moving_poisson.md",
    ],
  ],
  sitename="NonlinearROMs.jl",
  warnonly=[:cross_references,:missing_docs],
)

deploydocs(
  repo="github.com/nichomueller/NonlinearROMs.jl.git",
  push_preview=true,
)
