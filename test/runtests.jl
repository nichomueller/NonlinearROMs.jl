module NonlinearROMsTests

using Test

@testset "NonlinearROMs.jl" begin

  @testset "graphs" include("graphs.jl")
  @testset "data_interface" include("data_interface.jl")
  @testset "kernel_operators" include("kernel_operators.jl")
  @testset "poisson" include("poisson.jl")
  @testset "stokes" include("stokes.jl")
  @testset "moving poisson" include("moving_poisson.jl")
  @testset "heat equation" include("heat_equation.jl")

end

end # module