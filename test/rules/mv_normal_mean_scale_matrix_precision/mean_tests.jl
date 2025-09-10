
@testitem "rules:MvNormalMeanScaleMatrixPrecision:mean" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules

    @testset "Variational: (q_out::MultivariateNormalDistributionsFamily, q_γ::Gamma, q_G::Wishart)" begin
        q_out = MvNormalMeanPrecision([2.0, 3.0], [3.0 2.0; 2.0 4.0])
        q_γ = Gamma(3.0, 1.0)
        q_G = Wishart(4, [13.0 14.0; 14.0 20.0])
        MvNormalMeanPrecision(mean(q_out), mean(q_γ) * mean(q_G))
        @test_rules [check_type_promotion = true] MvNormalMeanScaleMatrixPrecision(:μ, Marginalisation) [
            (
                input = (q_out = MvNormalMeanCovariance([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_γ = GammaShapeRate(1.0, 1.0), q_G = Wishart(4, [13.0 14.0; 14.0 20.0])),
                output = MvNormalMeanPrecision([2.0, 1.0], [52.0 56.0; 56.0 80.0])
            ),
            (
                input = (q_out = MvNormalMeanPrecision([2.0, 3.0], [3.0 2.0; 2.0 4.0]), q_γ = Gamma(3.0, 1.0), q_G = Wishart(4, [13.0 14.0; 14.0 20.0])),
                output = MvNormalMeanPrecision([2.0, 3.0], [156.0 168.0; 168.0 240.0])
            )
        ]
    end

    @testset "Structured variational: (m_out::MultivariateNormalDistributionsFamily, q_γ::Gamma, q_G::Wishart)" begin
        # m_out = MvNormalMeanCovariance([0.0, 0.0], [7.0 -1.0; -1.0 9.0])
        # q_γ = GammaShapeRate(4.0, 2.0)
        # q_G = Wishart(4, [13.0 14.0; 14.0 20.0])
        # MvNormalMeanPrecision(mean(m_out), inv(cov(m_out) + inv(mean(q_γ) * mean(q_G))))
        @test_rules [check_type_promotion = true] MvNormalMeanScaleMatrixPrecision(:μ, Marginalisation) [
            (
                input = (m_out = MvNormalMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_γ = Gamma(1.0, 1.0), q_G = Wishart(4, [13.0 14.0; 14.0 20.0])),
                output = MvNormalMeanPrecision([2.0, 1.0], [2.777070063694267 1.9872611464968155; 1.9872611464968155 3.770700636942675])
            ),
            (
                input = (m_out = MvNormalMeanCovariance([0.0, 0.0], [7.0 -1.0; -1.0 9.0]), q_γ = GammaShapeRate(4.0, 2.0), q_G = Wishart(4, [13.0 14.0; 14.0 20.0])),
                output = MvNormalMeanPrecision([0.0, 0.0], [0.1444643743381126 0.016444116187372262; 0.016444116187372262 0.1126703322039727])
            ),
            (
                input = (m_out = MvNormalWeightedMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_γ = GammaShapeRate(2.0, 1.0), q_G = Wishart(4, [13.0 14.0; 14.0 20.0])),
                output = MvNormalMeanPrecision([0.7500000000000003, -0.12500000000000017], [2.8822495606326877 1.9964850615114234; 1.9964850615114234 3.8804920913884007])
            )
        ]
    end
end
