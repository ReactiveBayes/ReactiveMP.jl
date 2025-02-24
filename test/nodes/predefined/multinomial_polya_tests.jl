@testitem "MultinomialPolya average energy" begin
    using ReactiveMP, Random, BayesBase, ExponentialFamily, Distributions

    @testset "Average energy" begin
        let
            q_x = PointMass([10, 20, 70])  # K=3 categories
            q_N = PointMass(100)
            q_ψ = MvNormalWeightedMeanPrecision([0.5, 0.3], [1.0 0.2; 0.2 1.0])  # K-1=2 dimensions

            @test score(
                AverageEnergy(),
                MultinomialPolya,
                Val{(:x, :N, :ψ)}(),
                (Marginal(q_x, false, false, nothing), Marginal(q_N, false, false, nothing), Marginal(q_ψ, false, false, nothing)),
                nothing
            ) ≈ 104.19 atol = 0.1
        end

        let
            q_x = Multinomial(100, [0.2, 0.3, 0.5])
            q_N = PointMass(100)
            q_ψ = MvNormalMeanCovariance([0.1, -0.2], [2.0 0.5; 0.5 1.5])

            @test score(
                AverageEnergy(),
                MultinomialPolya,
                Val{(:x, :N, :ψ)}(),
                (Marginal(q_x, false, false, nothing), Marginal(q_N, false, false, nothing), Marginal(q_ψ, false, false, nothing)),
                nothing
            ) ≈ -101.72 atol = 0.1
        end

        let
            q_x = PointMass([30, 70])  # K=2
            q_N = PointMass(100)
            q_ψ = PointMass([0.5])

            result = score(
                AverageEnergy(),
                MultinomialPolya,
                Val{(:x, :N, :ψ)}(),
                (Marginal(q_x, false, false, nothing), Marginal(q_N, false, false, nothing), Marginal(q_ψ, false, false, nothing)),
                nothing
            )

            @test result ≈ 23.76 atol = 0.1
        end
    end
end
