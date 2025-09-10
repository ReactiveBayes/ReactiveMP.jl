
@testitem "rules:MvNormalMeanScaleMatrixPrecision:precision" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: GammaShapeRate, @test_rules

    @testset "Variational: (q_out::MultivariateNormalDistributionsFamily, q_μ::MultivariateNormalDistributionsFamily, q_G::Wishart)" begin
        # q_out = MvNormalMeanCovariance([1.0; 2.0], [3.0 2.0; 2.0 4.0])
        # q_μ = MvNormalMeanCovariance([3.0; 5.0], [3.0 2.0; 2.0 4.0])
        # q_G = Wishart(ndims(q_out) + 2, [13.0 14.0; 14.0 20.0])
        # output = GammaShapeRate(div(ndims(q_out), 2) + 1, 0.5*tr(mean(q_G)*(cov(q_out) + cov(q_μ) + (mean(q_out) - mean(q_μ))*(mean(q_out) - mean(q_μ))')))
        @test_rules [check_type_promotion = true] MvNormalMeanScaleMatrixPrecision(:γ, Marginalisation) [
            (
                input = (
                    q_out = MvNormalMeanCovariance([1.0; 2.0], [3.0 2.0; 2.0 4.0]),
                    q_μ = MvNormalMeanCovariance([3.0; 5.0], [3.0 2.0; 2.0 4.0]),
                    q_G = Wishart(4, [13.0 14.0; 14.0 20.0])
                ),
                output = GammaShapeRate(2.0, 1500.0)
            ),
            (
                input = (
                    q_out = MvNormalMeanPrecision([1.0; 2.0], [3.0 2.0; 2.0 4.0]),
                    q_μ = MvNormalMeanPrecision([3.0; 5.0], [3.0 2.0; 2.0 4.0]),
                    q_G = Wishart(4, [13.0 14.0; 14.0 20.0])
                ),
                output = GammaShapeRate(2.0, 828.0)
            )
        ]
    end

    @testset "Variational: (q_out_μ::MultivariateNormalDistributionsFamily, q_G::Wishart)" begin
        # a = [1.0, 2.0, -1.0, -2.0]
        # A = [3.5 -0.5 -0.25 0.0; -0.5 3.0 -0.25 0.0; -0.25 -0.25 6.0 0.25; 0.0 0.0 0.25 7.0]
        # q_out_μ = MvNormalWeightedMeanPrecision(a, A)
        # d = div(ndims(q_out_μ), 2)
        # q_G = Wishart(ndims(q_out) + 2, [13.0 14.0; 14.0 20.0])
        # m_out_μ, Cov_out_μ = mean_cov(q_out_μ)
        # m_out, m_μ = @views m_out_μ[1:d], m_out_μ[(d + 1):end]
        # Cov_out, Cov_μ = @views Cov_out_μ[1:d, 1:d], Cov_out_μ[(d + 1):end, (d + 1):end]
        # Cov_out_out, Cov_μ_μ = @views Cov_out_μ[1:d, (d + 1):end], Cov_out_μ[(d + 1):end, 1:d]
        # output = GammaShapeRate(d/2 + 1, 0.5*tr(mean(q_G)*(Cov_out + Cov_μ - Cov_out_out - Cov_μ_μ + (m_out - m_μ)*(m_out - m_μ)')))
        @test_rules [check_type_promotion = true] MvNormalMeanScaleMatrixPrecision(:γ, Marginalisation) [
            (input = (q_out_μ = MvNormalMeanCovariance(ones(4), diageye(4)), q_G = Wishart(4, [13.0 14.0; 14.0 20.0])), output = GammaShapeRate(2.0, 132.0)),
            (input = (q_out_μ = MvNormalMeanPrecision(ones(4), diageye(4)), q_G = Wishart(4, [13.0 14.0; 14.0 20.0])), output = GammaShapeRate(2.0, 132.0)),
            (input = (q_out_μ = MvNormalWeightedMeanPrecision(ones(4), diageye(4)), q_G = Wishart(4, [13.0 14.0; 14.0 20.0])), output = GammaShapeRate(2.0, 132.0))
        ]

        a = [1.0, 2.0, -1.0, -2.0]
        A = [3.5 -0.5 -0.25 0.0; -0.5 3.0 -0.25 0.0; -0.25 -0.25 6.0 0.25; 0.0 0.0 0.25 7.0]

        @test_rules [check_type_promotion = true] MvNormalMeanScaleMatrixPrecision(:γ, Marginalisation) [
            (input = (q_out_μ = MvNormalMeanCovariance(a, A), q_G = Wishart(4, [13.0 14.0; 14.0 20.0])), output = GammaShapeRate(2, 1852.0)),
            (input = (q_out_μ = MvNormalMeanPrecision(a, A), q_G = Wishart(4, [13.0 14.0; 14.0 20.0])), output = GammaShapeRate(2, 1224.3440167026847)),
            (input = (q_out_μ = MvNormalWeightedMeanPrecision(a, A), q_G = Wishart(4, [13.0 14.0; 14.0 20.0])), output = GammaShapeRate(2, 106.333391526103))
        ]
    end
end
