
@testitem "rules:MvNormalMeanScaleMatrixPrecision:matrix" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules

    @testset "Variational: (q_out::MultivariateNormalDistributionsFamily, q_μ::MultivariateNormalDistributionsFamily, q_γ::Gamma)" begin
        # q_out = MvNormalMeanCovariance([2.0; 5.0], [3.0 2.0; 2.0 4.0])
        # q_μ = MvNormalMeanCovariance([3.0; 5.0], [1.0 2.0; 2.0 3.0])
        # q_γ = GammaShapeRate(2.0, 4.0)
        # Wishart(4, inv((cov(q_out) + cov(q_μ) + (mean(q_out) - mean(q_μ))*(mean(q_out) - mean(q_μ))' ) * mean(q_γ)))
        @test_rules [check_type_promotion = true] MvNormalMeanScaleMatrixPrecision(:G, Marginalisation) [
            (
                input = (
                    q_out = MvNormalMeanCovariance([1.0; 2.0], [1.0 2.0; 2.0 3.0]), 
                    q_μ = MvNormalMeanCovariance([1.0; 2.0], [3.0 2.0; 2.0 4.0]), 
                    q_γ = GammaShapeRate(1.0, 1.0)
                    ),
                output = Wishart(4, [0.5833333333333333 -0.3333333333333333; -0.3333333333333333 0.3333333333333333])
            ),
            (
                input = (
                    q_out = MvNormalMeanCovariance([2.0; 5.0], [3.0 2.0; 2.0 4.0]), 
                    q_μ = MvNormalMeanCovariance([3.0; 5.0], [1.0 2.0; 2.0 3.0]), 
                    q_γ = GammaShapeRate(2.0, 4.0)
                    ),
                output = Wishart(4, [0.736842105263158 -0.42105263157894735; -0.42105263157894735 0.5263157894736842]))
        ]
    end

    @testset "Structured variational: (q_out_μ::MultivariateNormalDistributionsFamily, q_γ::Gamma)" begin
        # q_out_μ = MvNormalWeightedMeanPrecision(3*ones(4), diageye(4))
        # q_γ = GammaShapeRate(22.0, 14.0)
        # d=2
        # m_out_μ, Cov_out_μ = mean_cov(q_out_μ)
        # m_out, m_μ = @views m_out_μ[1:d], m_out_μ[(d + 1):end]
        # Cov_out, Cov_μ = @views Cov_out_μ[1:d, 1:d], Cov_out_μ[(d + 1):end, (d + 1):end]
        # Cov_out_out, Cov_μ_μ = @views Cov_out_μ[1:d, (d + 1):end], Cov_out_μ[(d + 1):end, 1:d]
        # Wishart(4, inv((Cov_out + Cov_μ - Cov_out_out - Cov_μ_μ + (m_out - m_μ)*(m_out - m_μ)' ) * mean(q_γ)))
        @test_rules [check_type_promotion = true] MvNormalMeanScaleMatrixPrecision(:G, Marginalisation) [
            (
                input = (
                    q_out_μ = MvNormalWeightedMeanPrecision(ones(4), diageye(4)),
                    q_γ = GammaShapeRate(1.0, 1.0)
                    ), 
                output = Wishart(4, [0.5 0.0; 0.0 0.5])
            ),
            (
                input = (
                    q_out_μ = MvNormalWeightedMeanPrecision(2*ones(4), diageye(4)),
                    q_γ = GammaShapeRate(2.0, 4.0)
                    ),
                output = Wishart(4, [1.0 0.0; 0.0 1.0])
            ),
            (
                input = (
                    q_out_μ = MvNormalWeightedMeanPrecision(3*ones(4), diageye(4)),
                    q_γ = GammaShapeRate(22.0, 14.0)
                    ),
                output = Wishart(4, [0.3181818181818182 0.0; 0.0 0.3181818181818182])
            )
        ]
    end
end
