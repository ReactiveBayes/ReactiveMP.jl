module RulesNormalAutoregressiveTest

using Test
using ReactiveMP
using Random
using Distributions

import ReactiveMP: @test_rules

@testset "rules:Autoregressive:x" begin
    @testset "Mean-field: (q_y::Any, q_θ::Any, q_γ::Any)" begin
        armeta = ARMeta(Univariate, 1, ARsafe())
        @test_rules [check_type_promotion = true] Autoregressive(:x, Marginalisation) [
            (
                input = (q_y = NormalMeanVariance(1.0, 1.0), q_θ = NormalMeanVariance(1.0, 1.0), q_γ = GammaShapeRate(1.0, 1.0), meta = armeta),
                output = NormalWeightedMeanPrecision(1.0, 2.0)
            ),
            (
                input = (q_y = NormalWeightedMeanPrecision(1.0, 1.0), q_θ = NormalMeanPrecision(1.0, 2.0), q_γ = GammaShapeScale(2.0, 1.0), meta = armeta),
                output = NormalWeightedMeanPrecision(2.0, 3.0)
            )
        ]
    end
    @testset "Mean-field: (q_y::Any, q_θ::Any, q_γ::Any)" begin
        order = 2
        armeta = ARMeta(Multivariate, order, ARsafe())
        @test_rules [check_type_promotion = true] Autoregressive(:x, Marginalisation) [
            (
                input = (
                    q_y = MvNormalMeanCovariance(zeros(order), diageye(order)),
                    q_θ = MvNormalMeanCovariance(ones(order), diageye(order)),
                    q_γ = GammaShapeRate(1.0, 1.0),
                    meta = armeta
                ),
                output = MvNormalWeightedMeanPrecision(zeros(2), [2.0 1.0; 1.0 2.0])
            ),
            (
                input = (
                    q_y = MvNormalMeanCovariance(ones(order), diageye(order)),
                    q_θ = MvNormalMeanCovariance(zeros(order), diageye(order)),
                    q_γ = GammaShapeScale(1.0, 1.0),
                    meta = armeta
                ),
                output = MvNormalWeightedMeanPrecision([0.0, 0.0], [1.0 0.0; 0.0 1.0])
            ),
            (
                input = (
                    q_y = MvNormalMeanCovariance(ones(order), diageye(order)), q_θ = MvNormalMeanCovariance(ones(order), diageye(order)), q_γ = Gamma(2.0, 1.0), meta = armeta
                ),
                output = MvNormalWeightedMeanPrecision([2.0, 2.0], [4.0 2.0; 2.0 4.0])
            )
        ]
    end

    @testset "Structured: (m_y::UnivariateNormalDistributionsFamily, q_θ::UnivariateNormalDistributionsFamily, q_γ::Any)" begin
        armeta = ARMeta(Univariate, 1, ARsafe())
        @test_rules [check_type_promotion = true] Autoregressive(:x, Marginalisation) [
            (
                input = (m_y = NormalMeanVariance(1.0, 1.0), q_θ = NormalMeanVariance(1.0, 1.0), q_γ = GammaShapeRate(1.0, 1.0), meta = armeta),
                output = NormalWeightedMeanPrecision(0.5, 1.5)
            ),
            (
                input = (m_y = NormalWeightedMeanPrecision(1.0, 1.0), q_θ = NormalMeanPrecision(1.0, 2.0), q_γ = GammaShapeScale(1.0, 1.0), meta = armeta),
                output = NormalWeightedMeanPrecision(0.5, 1.0)
            )
        ]
        # inconsistent inputs
        @test_throws MethodError @call_rule AR(:x, Marginalisation) (
            m_y = MvNormalMeanPrecision([0.0], [1.0;]), q_θ = NormalMeanPrecision(1.0, 1.0), q_γ = GammaShapeRate(1.0, 1.0), meta = armeta
        )
        # wrong meta specification
        armeta = ARMeta(Multivariate, 1, ARsafe())
        @test_throws MethodError @call_rule AR(:x, Marginalisation) (
            m_y = NormalMeanPrecision(0.0, 1.0), q_θ = NormalMeanPrecision(1.0, 1.0), q_γ = GammaShapeRate(1.0, 1.0), meta = armeta
        )
    end

    @testset "Structured : (m_y::MultivariateNormalDistributionsFamily, q_θ::MultivariateNormalDistributionsFamily, q_γ::Any)" begin
        order = 2
        armeta = ARMeta(Multivariate, order, ARsafe())
        @test_rules [check_type_promotion = true] Autoregressive(:x, Marginalisation) [
            (
                input = (
                    m_y = MvNormalMeanCovariance(zeros(order), diageye(order)),
                    q_θ = MvNormalMeanCovariance(ones(order), diageye(order)),
                    q_γ = GammaShapeRate(1.0, 1.0),
                    meta = armeta
                ),
                output = MvNormalWeightedMeanPrecision([0.0, 0.0], [2.5 0.5; 0.5 1.5])
            ),
            (
                input = (
                    m_y = MvNormalMeanCovariance(ones(order), diageye(order)),
                    q_θ = MvNormalMeanCovariance(zeros(order), diageye(order)),
                    q_γ = GammaShapeScale(1.0, 1.0),
                    meta = armeta
                ),
                output = MvNormalWeightedMeanPrecision([1.0, 0.0], [2.0 0.0; 0.0 1.0])
            ),
            (
                input = (
                    m_y = MvNormalMeanCovariance(ones(order), diageye(order)), q_θ = MvNormalMeanCovariance(ones(order), diageye(order)), q_γ = Gamma(1.0, 1.0), meta = armeta
                ),
                output = MvNormalWeightedMeanPrecision([1.5, 0.5], [2.5 0.5; 0.5 1.5])
            )
        ]
    end
end

end
