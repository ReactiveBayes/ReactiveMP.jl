
@testitem "rules:MvNormalMeanScaleMatrixPrecision:out" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules

    @testset "Variational: (q_μ::MultivariateNormalDistributionsFamily, q_γ::Gamma, q_G::Wishart)" begin
        # q_μ = MvNormalWeightedMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0])
        # q_γ = Gamma(4.0, 2.0)
        # q_G = Wishart(4, [13.0 14.0; 14.0 20.0])
        # MvNormalMeanPrecision(mean(q_μ), mean(q_γ) * mean(q_G))
        @test_rules [check_type_promotion = true] MvNormalMeanScaleMatrixPrecision(:out, Marginalisation) [
            (
                input = (q_μ = MvNormalMeanCovariance([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_γ = Gamma(1.0, 1.0), q_G = Wishart(4, [13.0 14.0; 14.0 20.0])),
                output = MvNormalMeanPrecision([2.0, 1.0], [52.0 56.0; 56.0 80.0])
            ),
            (
                input = (q_μ = MvNormalMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_γ = Gamma(3.0, 2.0), q_G = Wishart(4, [13.0 14.0; 14.0 20.0])),
                output = MvNormalMeanPrecision([2.0, 1.0], [312.0 336.0; 336.0 480.0])
            ),
            (
                input = (q_μ = MvNormalWeightedMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_γ = Gamma(4.0, 2.0), q_G = Wishart(4, [13.0 14.0; 14.0 20.0])),
                output = MvNormalMeanPrecision([0.7500000000000003, -0.12500000000000017], [416.0 448.0; 448.0 640.0])
            )
        ]
    end

    @testset "Structured variational: (m_μ::MultivariateNormalDistributionsFamily, q_γ::Gamma, q_G::Wishart)" begin
        m_μ = MvNormalMeanCovariance([2.0, 1.0], [3.0 2.0; 2.0 4.0])
        q_γ = Gamma(2.0, 1.0)
        q_G = Wishart(4, [13.0 14.0; 14.0 20.0])
        MvNormalMeanPrecision(mean(m_μ), inv(cov(m_μ) + inv(mean(q_γ) * mean(q_G))))
        @test_rules [check_type_promotion = true] MvNormalMeanScaleMatrixPrecision(:out, Marginalisation) [
            (
                input = (m_μ = MvNormalMeanCovariance([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_γ = Gamma(2.0, 1.0), q_G = Wishart(4, [13.0 14.0; 14.0 20.0])),
                output = MvNormalMeanPrecision([2.0, 1.0], [0.4825426556235183 -0.23647165559425204; -0.23647165559425204 0.3643068278263923])
            ),
            (
                input = (m_μ = MvNormalMeanCovariance([0.0, 1.0], [2.0 -1.0; -1.0 4.0]), q_γ = Gamma(3.0, 1.0), q_G = Wishart(4, [13.0 14.0; 14.0 20.0])),
                output = MvNormalMeanPrecision([0.0, 1.0], [0.5656312548702044 0.14337881403841166; 0.14337881403841166 0.2852908371403689])
            ),
            (
                input = (m_μ = MvNormalWeightedMeanPrecision([3.0, -1.0], [1.0 0.0; 0.0 1.0]), q_γ = Gamma(4.0, 2.0), q_G = Wishart(4, [13.0 14.0; 14.0 20.0])),
                output = MvNormalMeanPrecision([3.0, -1.0], [0.9903743636718574 0.006727433814364876; 0.006727433814364877 0.9937380805790399])
            )
        ]
    end
end
