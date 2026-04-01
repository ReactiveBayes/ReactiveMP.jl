
@testitem "ProbitNode" begin
    using ReactiveMP, Random, BayesBase, ExponentialFamily

    @testset "Average energy" begin
        @test score(
            AverageEnergy(),
            Probit,
            Val{(:out, :in)}(),
            (
                Marginal(Bernoulli(1), false, false),
                Marginal(NormalMeanVariance(0.0, 1.0), false, false),
            ),
            ProbitMeta(),
        ) ≈ 1.0

        @test score(
            AverageEnergy(),
            Probit,
            Val{(:out, :in)}(),
            (
                Marginal(PointMass(1), false, false),
                Marginal(NormalMeanVariance(0.0, 1.0), false, false),
            ),
            ProbitMeta(100),
        ) ≈ 1.0

        for k in 0:0.1:1
            @test score(
                AverageEnergy(),
                Probit,
                Val{(:out, :in)}(),
                (
                    Marginal(Bernoulli(k), false, false),
                    Marginal(NormalMeanVariance(0.0, 1.0), false, false),
                ),
                ProbitMeta(100),
            ) ≈ 1.0
        end
    end
end
