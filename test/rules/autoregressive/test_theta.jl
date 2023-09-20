module RulesNormalAutoregressiveTest

using Test
using ReactiveMP
using Random
using Distributions

import ReactiveMP: @test_rules

@testset "rules:Autoregressive:θ" begin
    @testset "Mean-field: (q_x::Any, q_y::Any, q_γ::Any)" begin
        armeta = ARMeta(Univariate, 1, ARsafe())
        @test_rules [check_type_promotion = true] Autoregressive(:θ, Marginalisation) [
            (
                input = (q_y = NormalMeanVariance(1.0, 1.0), q_x = NormalMeanVariance(1.0, 1.0), q_γ = GammaShapeRate(1.0, 1.0), meta = armeta),
                output = NormalWeightedMeanPrecision(1.0, 2.0)
            ),
            (
                input = (q_y = NormalWeightedMeanPrecision(1.0, 1.0), q_x = NormalMeanPrecision(1.0, 2.0), q_γ = GammaShapeScale(2.0, 1.0), meta = armeta),
                output = NormalWeightedMeanPrecision(2.0, 3.0)
            )
        ]
    end
    @testset "Mean-field: (q_y::Any, q_θ::Any, q_γ::Any)" begin
        order = 2
        armeta = ARMeta(Multivariate, order, ARsafe())
        @test_rules [check_type_promotion = true] Autoregressive(:θ, Marginalisation) [
            (
                input = (
                    q_y = MvNormalMeanCovariance(zeros(order), diageye(order)),
                    q_x = MvNormalMeanCovariance(ones(order), diageye(order)),
                    q_γ = GammaShapeRate(1.0, 1.0),
                    meta = armeta
                ),
                output = MvNormalWeightedMeanPrecision(zeros(2), [2.0 1.0; 1.0 2.0])
            ),
            (
                input = (
                    q_y = MvNormalMeanCovariance(ones(order), diageye(order)),
                    q_x = MvNormalMeanCovariance(zeros(order), diageye(order)),
                    q_γ = GammaShapeScale(1.0, 1.0),
                    meta = armeta
                ),
                output = MvNormalWeightedMeanPrecision([0.0, 0.0], [1.0 0.0; 0.0 1.0])
            ),
            (
                input = (
                    q_y = MvNormalMeanCovariance(ones(order), diageye(order)), q_x = MvNormalMeanCovariance(ones(order), diageye(order)), q_γ = Gamma(2.0, 1.0), meta = armeta
                ),
                output = MvNormalWeightedMeanPrecision([2.0, 2.0], [4.0 2.0; 2.0 4.0])
            )
        ]
    end

    @testset "Structured: (q_y_x::MultivariateNormalDistributionsFamily, q_γ::Any)" begin
        armeta = ARMeta(Univariate, 1, ARsafe())
        @test_rules [check_type_promotion = true] Autoregressive(:θ, Marginalisation) [
            (input = (q_y_x = MvNormalMeanCovariance(ones(2), diageye(2)), q_γ = GammaShapeRate(1.0, 1.0), meta = armeta), output = NormalWeightedMeanPrecision(1.0, 2.0)),
            (input = (q_y_x = MvNormalMeanCovariance(2 * ones(2), diageye(2)), q_γ = GammaShapeScale(2.0, 1.0), meta = armeta), output = NormalWeightedMeanPrecision(8.0, 10.0))
        ]
    end

    @testset "Structured : (q_y_x::MultivariateNormalDistributionsFamily, q_γ::Any)" begin
        order = 2
        armeta = ARMeta(Multivariate, order, ARsafe())
        @test_rules [check_type_promotion = true] Autoregressive(:θ, Marginalisation) [
            (
                input = (q_y_x = MvNormalMeanCovariance(ones(2 * order), diageye(2 * order)), q_γ = GammaShapeRate(1.0, 1.0), meta = armeta),
                output = MvNormalWeightedMeanPrecision(ones(order), [2.0 1.0; 1.0 2.0])
            ),
            (
                input = (q_y_x = MvNormalMeanCovariance(zeros(2 * order), diageye(2 * order)), q_γ = GammaShapeRate(1.0, 1.0), meta = armeta),
                output = MvNormalWeightedMeanPrecision(zeros(order), [1.0 0.0; 0.0 1.0])
            )
        ]
    end
end

end
