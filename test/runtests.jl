module ReactiveMPTest

using Test, Documenter, ReactiveMP

# doctest(ReactiveMP)

@testset "ReactiveMP" begin

    include("test_distributions.jl")
    include("distributions/test_normal_mean_variance.jl")
    include("distributions/test_normal_mean_precision.jl")
    include("distributions/test_mv_normal_mean_precision.jl")

    include("test_node.jl")

    @testset "Detect ambiguities" begin
        @test length(Test.detect_ambiguities(Rocket)) == 0
    end
end

end