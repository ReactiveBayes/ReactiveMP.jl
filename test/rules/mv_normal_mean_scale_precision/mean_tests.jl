
@testitem "rules:MvNormalMeanScalePrecision:mean" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules

    @testset "Variational: (q_out::MultivariateNormalDistributionsFamily, q_γ::Gamma)" begin
        @test_rules [check_type_promotion = true] MvNormalMeanScalePrecision(:μ, Marginalisation) [
            (
                input = (q_out = MvNormalMeanCovariance([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_γ = GammaShapeRate(1.0, 1.0)),
                output = MvNormalMeanPrecision([2.0, 1.0], [1.0 0.0; 0.0 1.0])
            ),
            (input = (q_out = MvNormalMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_γ = Gamma(3.0, 1.0)), output = MvNormalMeanPrecision([2.0, 1.0], [3.0 0.0; 0.0 3.0]))
        ]
    end

    @testset "Structured variational: (m_out::MultivariateNormalDistributionsFamily, q_γ::Gamma)" begin
        @test_rules [check_type_promotion = true] MvNormalMeanScalePrecision(:μ, Marginalisation) [
            (input = (m_out = MvNormalMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_γ = Gamma(1.0, 1.0)), output = MvNormalMeanCovariance([2.0, 1.0], [1.5 -0.25; -0.25 1.375])),
            (
                input = (m_out = MvNormalMeanCovariance([0.0, 0.0], [7.0 -1.0; -1.0 9.0]), q_γ = GammaShapeRate(4.0, 2.0)),
                output = MvNormalMeanCovariance([0.0, 0.0], [7.5 -1.0; -1.0 9.5])
            ),
            (
                input = (m_out = MvNormalWeightedMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_γ = GammaShapeRate(2.0, 1.0)),
                output = MvNormalMeanCovariance([3 / 4, -1 / 8], [1.0 -0.25; -0.25 0.875])
            )
        ]
    end
end
