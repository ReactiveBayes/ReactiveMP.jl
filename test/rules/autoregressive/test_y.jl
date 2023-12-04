module RulesNormalAutoregressiveTest

using Test, ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

import ReactiveMP: @test_rules, ARTransitionMatrix

@testset "rules:Autoregressive:y" begin
    @testset "Mean-field: (q_x::Any, q_θ::Any, q_γ::Any)" begin
        armeta = ARMeta(Univariate, 1, ARsafe())
        @test_rules [check_type_promotion = true] Autoregressive(:y, Marginalisation) [
            (
                input = (q_x = NormalMeanVariance(1.0, 1.0), q_θ = NormalMeanVariance(1.0, 1.0), q_γ = GammaShapeRate(1.0, 1.0), meta = armeta),
                output = NormalMeanVariance(1.0, 1.0)
            ),
            (
                input = (q_x = NormalWeightedMeanPrecision(1.0, 1.0), q_θ = NormalMeanPrecision(1.0, 2.0), q_γ = GammaShapeScale(2.0, 1.0), meta = armeta),
                output = NormalMeanVariance(1.0, 0.5)
            )
        ]
    end
    @testset "Mean-field: (q_x::Any, q_θ::Any, q_γ::Any)" begin
        order = 2
        armeta = ARMeta(Multivariate, order, ARsafe())
        @test_rules [check_type_promotion = true] Autoregressive(:y, Marginalisation) [
            (
                input = (
                    q_x = MvNormalMeanCovariance(zeros(order), diageye(order)),
                    q_θ = MvNormalMeanCovariance(ones(order), diageye(order)),
                    q_γ = GammaShapeRate(1.0, 1.0),
                    meta = armeta
                ),
                output = MvNormalMeanCovariance(zeros(2), ARTransitionMatrix(order, 1.0))
            ),
            (
                input = (
                    q_x = MvNormalMeanCovariance(ones(order), diageye(order)),
                    q_θ = MvNormalMeanCovariance(zeros(order), diageye(order)),
                    q_γ = GammaShapeScale(1.0, 1.0),
                    meta = armeta
                ),
                output = MvNormalMeanCovariance([0.0, 1.0], ARTransitionMatrix(order, 1.0))
            ),
            (
                input = (
                    q_x = MvNormalMeanCovariance(ones(order), diageye(order)), q_θ = MvNormalMeanCovariance(ones(order), diageye(order)), q_γ = Gamma(2.0, 1.0), meta = armeta
                ),
                output = MvNormalMeanCovariance([2.0, 1.0], ARTransitionMatrix(order, 2.0))
            )
        ]
    end

    @testset "Structured: (m_x::UnivariateNormalDistributionsFamily, q_θ::UnivariateNormalDistributionsFamily, q_γ::Any)" begin
        armeta = ARMeta(Univariate, 1, ARsafe())
        @test_rules [check_type_promotion = true] Autoregressive(:y, Marginalisation) [
            (
                input = (m_x = NormalMeanVariance(1.0, 1.0), q_θ = NormalMeanVariance(1.0, 1.0), q_γ = GammaShapeRate(1.0, 1.0), meta = armeta),
                output = NormalMeanVariance(0.5, 1.5)
            ),
            (
                input = (m_x = NormalWeightedMeanPrecision(1.0, 1.0), q_θ = NormalMeanPrecision(1.0, 2.0), q_γ = GammaShapeScale(2.0, 1.0), meta = armeta),
                output = NormalMeanVariance(0.5, 1.0)
            )
        ]
        # inconsistent inputs
        @test_throws MethodError @call_rule AR(:y, Marginalisation) (
            m_x = MvNormalMeanPrecision([0.0], [1.0;]), q_θ = NormalMeanPrecision(1.0, 1.0), q_γ = GammaShapeRate(1.0, 1.0), meta = armeta
        )
        # wrong meta specification
        armeta = ARMeta(Multivariate, 1, ARsafe())
        @test_throws MethodError @call_rule AR(:y, Marginalisation) (
            m_x = NormalMeanPrecision(0.0, 1.0), q_θ = NormalMeanPrecision(1.0, 1.0), q_γ = GammaShapeRate(1.0, 1.0), meta = armeta
        )
    end

    @testset "Structured : (m_x::MultivariateNormalDistributionsFamily, q_θ::MultivariateNormalDistributionsFamily, q_γ::Any)" begin
        order = 2
        armeta = ARMeta(Multivariate, order, ARsafe())
        @test_rules [check_type_promotion = true] Autoregressive(:y, Marginalisation) [
            (
                input = (
                    m_x = MvNormalMeanCovariance(zeros(order), diageye(order)),
                    q_θ = MvNormalMeanCovariance(ones(order), diageye(order)),
                    q_γ = GammaShapeRate(1.0, 1.0),
                    meta = armeta
                ),
                output = MvNormalMeanCovariance([0.0, 0.0], [2.0 0.5; 0.5 0.5])
            ),
            (
                input = (
                    m_x = MvNormalMeanCovariance(ones(order), diageye(order)),
                    q_θ = MvNormalMeanCovariance(zeros(order), diageye(order)),
                    q_γ = GammaShapeScale(1.0, 1.0),
                    meta = armeta
                ),
                output = MvNormalMeanCovariance([0.0, 0.5], [1.0 0.0; 0.0 0.5])
            ),
            (
                input = (
                    m_x = MvNormalMeanCovariance(ones(order), diageye(order)), q_θ = MvNormalMeanCovariance(ones(order), diageye(order)), q_γ = Gamma(1.0, 1.0), meta = armeta
                ),
                output = MvNormalMeanCovariance([1.0, 0.5], [2.0 0.5; 0.5 0.5])
            )
        ]
    end
end

end
