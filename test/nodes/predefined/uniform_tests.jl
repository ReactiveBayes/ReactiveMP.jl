
@testitem "UniformNode" begin
    using ReactiveMP, Random, BayesBase, ExponentialFamily, Distributions

    @testset "Average energy" begin
        a, b = 0.0, 1.0
        α, β = rand(0.1:0.1:1.0), rand(0.1:0.1:1.0)
        @test score(
            AverageEnergy(),
            Uniform,
            Val{(:out, :a, :b)}(),
            (Marginal(Beta(α, β), false, false, nothing), Marginal(PointMass(a), false, false, nothing), Marginal(PointMass(b), false, false, nothing)),
            nothing
        ) == 0.0
    end
end
