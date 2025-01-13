@testitem "BinomialPolya" begin
    using ReactiveMP, Random, BayesBase, ExponentialFamily

    @testset "Average energy" begin
        @test score(
            AverageEnergy(),
            BinomialPolya,
            Val{(:y, :x, :n, :β)}(),
            (Marginal(PointMass(1), false, false, nothing), Marginal(PointMass([0.1, 0.2]), false, false, nothing), Marginal(PointMass(5), false, false, nothing), Marginal(MvNormalWeightedMeanPrecision(zeros(2), diageye(2)), false, false, nothing)),
            nothing
        ) ≈ 1.856 atol = 1e-3
    end
end
