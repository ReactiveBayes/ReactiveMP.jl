module RulesNormalAutoregressiveTest

using Test, ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

import ReactiveMP: @test_rules

@testset "rules:Autoregressive:γ" begin
    @testset "Mean-field: (q_x::Any, q_y::Any, q_θ::Any)" begin
        armeta = ARMeta(Univariate, 1, ARsafe())
        @test_rules [check_type_promotion = true] Autoregressive(:γ, Marginalisation) [
            (
                input = (q_y = NormalMeanVariance(1.0, 1.0), q_x = NormalMeanVariance(1.0, 1.0), q_θ = NormalMeanPrecision(1.0, 1.0), meta = armeta),
                output = GammaShapeRate(3 / 2, 3 / 2)
            ),
            (
                input = (q_y = NormalWeightedMeanPrecision(1.0, 1.0), q_x = NormalMeanPrecision(1.0, 2.0), q_θ = NormalMeanPrecision(2.0, 1.0), meta = armeta),
                output = GammaShapeRate(3 / 2, 5 / 2)
            )
        ]
    end
    @testset "Mean-field: (q_y::Any, q_θ::Any, q_θ::Any)" begin
        order = 2
        armeta = ARMeta(Multivariate, order, ARsafe())
        @test_rules [check_type_promotion = true] Autoregressive(:γ, Marginalisation) [
            (
                input = (
                    q_y = MvNormalMeanCovariance(zeros(order), diageye(order)),
                    q_x = MvNormalMeanCovariance(ones(order), diageye(order)),
                    q_θ = MvNormalMeanCovariance(ones(order), diageye(order)),
                    meta = armeta
                ),
                output = GammaShapeRate(3 / 2, 9 / 2)
            ),
            (
                input = (
                    q_y = MvNormalMeanCovariance(ones(order), diageye(order)),
                    q_x = MvNormalMeanCovariance(zeros(order), diageye(order)),
                    q_θ = MvNormalMeanCovariance(ones(order), diageye(order)),
                    meta = armeta
                ),
                output = GammaShapeRate(3 / 2, 2.0)
            ),
            (
                input = (
                    q_y = MvNormalMeanCovariance(ones(order), diageye(order)),
                    q_x = MvNormalMeanCovariance(ones(order), diageye(order)),
                    q_θ = MvNormalMeanCovariance(ones(order), diageye(order)),
                    meta = armeta
                ),
                output = GammaShapeRate(3 / 2, 3.0)
            )
        ]
    end

    @testset "Structured: (q_y_x::MultivariateNormalDistributionsFamily, q_θ::Any)" begin
        armeta = ARMeta(Univariate, 1, ARsafe())
        @test_rules [check_type_promotion = true] Autoregressive(:γ, Marginalisation) [
            (input = (q_y_x = MvNormalMeanCovariance(ones(2), diageye(2)), q_θ = NormalMeanPrecision(1.0, 1.0), meta = armeta), output = GammaShapeRate(3 / 2, 2.0)),
            (input = (q_y_x = MvNormalMeanCovariance(2 * ones(2), diageye(2)), q_θ = NormalMeanPrecision(2.0, 1.0), meta = armeta), output = GammaShapeRate(3 / 2, 7.0))
        ]
    end

    @testset "Structured : (q_y_x::MultivariateNormalDistributionsFamily, q_θ::Any)" begin
        order = 2
        armeta = ARMeta(Multivariate, order, ARsafe())
        @test_rules [check_type_promotion = true] Autoregressive(:γ, Marginalisation) [
            (
                input = (q_y_x = MvNormalMeanCovariance(ones(2 * order), diageye(2 * order)), q_θ = MvNormalMeanPrecision(ones(order), diageye(order)), meta = armeta),
                output = GammaShapeRate(3 / 2, 4.0)
            ),
            (
                input = (q_y_x = MvNormalMeanCovariance(ones(2 * order), diageye(2 * order)), q_θ = MvNormalMeanPrecision(zeros(order), diageye(order)), meta = armeta),
                output = GammaShapeRate(3 / 2, 3.0)
            )
        ]
    end
end

end
