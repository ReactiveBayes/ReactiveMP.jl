module RulesMvNormalMeanScalePrecisionPrecisionTest

using Test
using ReactiveMP
using Random

import ReactiveMP: GammaShapeRate, @test_rules

@testset "rules:MvNormalMeanScalePrecision:precision" begin
    @testset "Variational: (q_out::MultivariateNormalDistributionsFamily, q_μ::MultivariateNormalDistributionsFamily)" begin
        @test_rules [with_float_conversions = false, float32_atol = 1e-5] MvNormalMeanScalePrecision(
            :γ,
            Marginalisation
        ) [
            (
                input = (
                    q_out = MvNormalMeanCovariance([1.0; 2.0], [3.0 2.0; 2.0 4.0]),
                    q_μ = MvNormalMeanCovariance([3.0; 5.0], [3.0 2.0; 2.0 4.0])
                ),
                output = GammaShapeRate(2.0, 13.5)
            ),
            (
                input = (
                    q_out = MvNormalMeanPrecision([1.0; 2.0], [3.0 2.0; 2.0 4.0]),
                    q_μ = MvNormalMeanPrecision([3.0; 5.0], [3.0 2.0; 2.0 4.0])
                ),
                output = GammaShapeRate(2.0, 7.375)
            )
        ]
    end

    @testset "Variational: (q_out_μ::MultivariateNormalDistributionsFamily)" begin
        @test_rules [with_float_conversions = false] MvNormalMeanScalePrecision(:γ, Marginalisation) [
            (
                input = (q_out_μ = MvNormalMeanCovariance(ones(4), diageye(4)),),
                output = GammaShapeRate(2.0, 2.0)
            ),
            (
                input = (q_out_μ = MvNormalMeanPrecision(ones(4), diageye(4)),),
                output = GammaShapeRate(2.0, 2.0)
            ),
            (
                input = (q_out_μ = MvNormalWeightedMeanPrecision(ones(4), diageye(4)),),
                output = GammaShapeRate(2.0, 2.0)
            )
        ]

        a = [1.0, 2.0, -1.0, -2.0]
        A = [3.5 -0.5 -0.25 0.0; -0.5 3.0 -0.25 0.0; -0.25 -0.25 6.0 0.25; 0.0 0.0 0.25 7.0]

        @test_rules [with_float_conversions = false, float32_atol = 1e-5] MvNormalMeanScalePrecision(
            :γ,
            Marginalisation
        ) [
            (
                input = (q_out_μ = MvNormalMeanCovariance(a, A),),
                output = GammaShapeRate(2, 20)
            ),
            (
                input = (q_out_μ = MvNormalMeanPrecision(a, A),),
                output = GammaShapeRate(2, 10.460349437749533)
            ),
            (
                input = (q_out_μ = MvNormalWeightedMeanPrecision(a, A),),
                output = GammaShapeRate(2, 1.0832705501813922)
            )
        ]
    end
end

end
