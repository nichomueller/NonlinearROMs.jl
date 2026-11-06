module NonlinearROMsTests

using Test

@testset "NonlinearROMs.jl" begin

  @testset "graphs" begin
    include("graphs.jl")
  end

  @testset "data_interface" begin
    include("data_interface.jl")
  end

  @testset "kernel_operators" begin
    include("kernel_operators.jl")
  end

  @testset "poisson" begin
    include("poisson.jl")
  end

end

end # module