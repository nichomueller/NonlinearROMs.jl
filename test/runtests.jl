module NonlinearROMsTests

using Test

@testset "poisson" begin include("poisson.jl") end
@testset "graphs" begin include("graphs.jl") end
@testset "neural operators helpers" begin include("NeuralOperators/neural_helpers.jl") end
@testset "deeponet pipeline" begin include("NeuralOperators/deeponet_pipeline.jl") end
@testset "nomad pipeline" begin include("NeuralOperators/nomad_pipeline.jl") end
@testset "kernel integral framework" begin include("kernel_integral_framework.jl") end

end # module
