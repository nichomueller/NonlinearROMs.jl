using Documenter
using Literate
using NonlinearROMs

Literate.markdown(
  joinpath(@__DIR__,"literate","poisson.jl"),
  joinpath(@__DIR__,"src");
  name="example",
  documenter=true,
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
    ],
  ],
  sitename="NonlinearROMs.jl",
  warnonly=[:cross_references,:missing_docs],
)

deploydocs(
  repo="github.com/nichomueller/NonlinearROMs.jl.git",
  push_preview=true,
)
