module NonlinearROMsTests

using Test

# every script runs in its own fresh module, as it would when run on its own:
# the scripts define globals (and constants) with the same names, e.g. `L`, `N`
runtest(file) = Base.include(Module(Symbol(splitext(file)[1])),joinpath(@__DIR__,file))

@testset "graphs" begin runtest("graphs.jl") end
@testset "data_interface" begin runtest("data_interface.jl") end
@testset "kernel_operators" begin runtest("kernel_operators.jl") end
@testset "poisson" begin runtest("poisson.jl") end
@testset "stokes" begin runtest("stokes.jl") end
@testset "moving poisson" begin runtest("moving_poisson.jl") end
@testset "heat equation" begin runtest("heat_equation.jl") end

end # module
