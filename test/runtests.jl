module NonlinearROMsTests

using Test

@testset "poisson" begin include("poisson.jl") end
@testset "graphs" begin include("graphs.jl") end

end # module
