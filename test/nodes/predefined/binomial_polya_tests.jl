@testitem "BinomialPolya average energy" begin
    using ReactiveMP, Random, BayesBase, ExponentialFamily

    @testset "Average energy" begin
        # Test with default meta (nothing)
        @test score(
            AverageEnergy(),
            BinomialPolya,
            Val{(:y, :x, :n, :β)}(),
            (
                Marginal(PointMass(1), false, false),
                Marginal(PointMass([0.1, 0.2]), false, false),
                Marginal(PointMass(5), false, false),
                Marginal(
                    MvNormalWeightedMeanPrecision(zeros(2), diageye(2)),
                    false,
                    false,
                ),
            ),
            nothing,
        ) ≈ 1.856 atol = 1e-3

        # Test with meta and fixed RNG
        rng = MersenneTwister(123)
        meta = BinomialPolyaMeta(1, rng)
        @test score(
            AverageEnergy(),
            BinomialPolya,
            Val{(:y, :x, :n, :β)}(),
            (
                Marginal(PointMass(1), false, false),
                Marginal(PointMass([0.1, 0.2]), false, false),
                Marginal(PointMass(5), false, false),
                Marginal(
                    MvNormalWeightedMeanPrecision(zeros(2), diageye(2)),
                    false,
                    false,
                ),
            ),
            meta,
        ) ≈ 1.856 atol = 1e-3
    end
end
