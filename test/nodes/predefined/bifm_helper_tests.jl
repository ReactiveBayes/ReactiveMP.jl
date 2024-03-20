
@testitem "BIFMHelperNode" begin
    using ReactiveMP, Random, ExponentialFamily, BayesBase

    @testset "Average energy" begin

        @test score(
            AverageEnergy(),
            BIFMHelper,
            Val{(:out, :in)}(),
            (Marginal(MvNormalMeanCovariance([1, 1], [2 0; 0 3]), false, false, nothing), Marginal(MvNormalMeanCovariance([1, 1], [2 0; 0 3]), false, false, nothing)),
            nothing
        ) ≈ entropy(MvNormalMeanCovariance([1, 1], [2 0; 0 3]))

        @test score(
            AverageEnergy(),
            BIFMHelper,
            Val{(:out, :in)}(),
            (Marginal(MvNormalMeanCovariance([1, 2], [2 0; 0 1]), false, false, nothing), Marginal(MvNormalMeanPrecision([1, 2], [0.5 0; 0 1]), false, false, nothing)),
            nothing
        ) ≈ entropy(MvNormalMeanCovariance([1, 2], [2 0; 0 1]))
    end
end
