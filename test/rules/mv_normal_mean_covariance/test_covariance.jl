module RulesMvNormalMeanCovarianceCovarianceTest

using Test
using ReactiveMP
using Random

import ReactiveMP: @test_rules
#TODO:
@testset "rules:MvNormalMeanCovariance:covariance" begin
    @testset "Variational: (q_out::PointMass, q_μ::MultivariateNormalDistributionsFamily)" begin
        @test_rules [with_float_conversions = false, float32_atol = 1e-5] MvNormalMeanCovariance(:Σ, Marginalisation) [
            (
                input = (q_out = PointMass([1.0, 2.0]), q_μ = MvNormalMeanCovariance([3.0, 5.0], [3.0 2.0; 2.0 4.0])),
                output = InvWishart(-2.0, [7.0 8.0; 8.0 13.0])
            ),
            (
                input = (q_out = MvNormalMeanCovariance([1.0; 2.0], [3.0 2.0; 2.0 4.0]), q_μ = PointMass([3.0; 5.0])),
                output = InvWishart(-2.0, [7.0 8.0; 8.0 13.0])
            ),
            (
                input = (
                    q_out = MvNormalMeanCovariance([1.0; 2.0], [3.0 2.0; 2.0 4.0]),
                    q_μ = MvNormalMeanCovariance([3.0; 5.0], [3.0 2.0; 2.0 4.0])
                ),
                output = InvWishart(-2.0, [10.0 10.0; 10.0 17.0])
            ),
            (
                input = (
                    q_out = MvNormalMeanCovariance([1.0; 2.0; 3.0], [3.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 4.0]),
                    q_μ = MvNormalMeanCovariance([3.0; 5.0; -2.0], [3.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 4.0])
                ),
                output = InvWishart(-3.0, [10.0 6.0 -10.0; 6.0 13.0 -15.0; -10.0 -15.0 33.0])
            )
        ]
        @test_rules [with_float_conversions = false, float32_atol = 1e-5] MvNormalMeanCovariance(:Σ, Marginalisation) [
            (
                input = (q_out = PointMass([1.0, 2.0]), q_μ = MvNormalMeanPrecision([3.0, 5.0], [3.0 2.0; 2.0 4.0])),
                output = InvWishart(-2.0, [9/2 23/4; 23/4 75/8])
            ),
            (
                input = (q_out = MvNormalMeanPrecision([1.0; 2.0], [3.0 2.0; 2.0 4.0]), q_μ = PointMass([3.0; 5.0])),
                output = InvWishart(-2.0, [9/2 23/4; 23/4 75/8])
            ),
            (
                input = (
                    q_out = MvNormalMeanPrecision([1.0; 2.0], [3.0 2.0; 2.0 4.0]),
                    q_μ = MvNormalMeanPrecision([3.0; 5.0], [3.0 2.0; 2.0 4.0])
                ),
                output = InvWishart(-2.0, [10/2 11/2; 11/2 39/4])
            ),
            (
                input = (
                    q_out = MvNormalMeanPrecision([1.0; 2.0; 3.0], [3.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 4.0]),
                    q_μ = MvNormalMeanPrecision([3.0; 5.0; -2.0], [3.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 4.0])
                ),
                output = InvWishart(-3.0, [14/3 6.0 -10.0; 6.0 10.0 -15.0; -10.0 -15.0 51/2])
            )
        ]
        @test_rules [with_float_conversions = false, float32_atol = 1e-5] MvNormalMeanCovariance(:Σ, Marginalisation) [
            (
                input = (
                    q_out = PointMass([1.0, 2.0]),
                    q_μ = MvNormalWeightedMeanPrecision([3.0, 5.0], [3.0 2.0; 2.0 4.0])
                ),
                output = InvWishart(-2.0, [17//16 13//32; 13//32 73//64])
            ),
            (
                input = (
                    q_out = MvNormalWeightedMeanPrecision([1.0; 2.0], [3.0 2.0; 2.0 4.0]),
                    q_μ = PointMass([3.0; 5.0])
                ),
                output = InvWishart(-2.0, [19/2 53/4; 53/4 165/8])
            ),
            (
                input = (
                    q_out = MvNormalWeightedMeanPrecision([1.0; 2.0], [3.0 2.0; 2.0 4.0]),
                    q_μ = MvNormalWeightedMeanPrecision([3.0; 5.0], [3.0 2.0; 2.0 4.0])
                ),
                output = InvWishart(-2.0, [17/16 -11/32; -11/32 73/64])
            )
        ]
    end

    @testset "Variational: (q_out_μ::MultivariateNormalDistributionsFamily)" begin
        @test_rules [with_float_conversions = false] MvNormalMeanCovariance(:Σ, Marginalisation) [
            (
                input = (q_out_μ = MvNormalMeanCovariance(ones(4), diageye(4)),),
                output = InvWishart(-2.0, [2.0 0.0; 0.0 2.0])
            ),
            (
                input = (q_out_μ = MvNormalMeanPrecision(ones(4), diageye(4)),),
                output = InvWishart(-2.0, [2.0 0.0; 0.0 2.0])
            ),
            (
                input = (q_out_μ = MvNormalWeightedMeanPrecision(ones(4), diageye(4)),),
                output = InvWishart(-2.0, [2.0 0.0; 0.0 2.0])
            )
        ]

        a = [1.0, 2.0, -1.0, -2.0]
        A = [3.5 -0.5 -0.25 0.0; -0.5 3.0 -0.25 0.0; -0.25 -0.25 6.0 0.25; 0.0 0.0 0.25 7.0]

        @test_rules [with_float_conversions = false, float32_atol = 1e-5] MvNormalMeanCovariance(:Σ, Marginalisation) [
            (input = (q_out_μ = MvNormalMeanCovariance(a, A),), output = InvWishart(-2.0, [14.0 8.0; 8.0 26.0]))
        ]
    end
end

end
