module RulesMvNormalMeanScalePrecisionOutTest

using Test
using ReactiveMP
using Random

import ReactiveMP: @test_rules

@testset "rules:MvNormalMeanPrecision:out" begin
    @testset "Variational: (q_out::MultivariateNormalDistributionsFamily, q_γ::Gamma)" begin
        @test_rules [with_float_conversions = true] MvNormalMeanScalePrecision(:out, Marginalisation) [
            (input = (q_μ = MvNormalMeanCovariance([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_γ = Gamma(1.0, 1.0)), output = MvNormalMeanPrecision([2.0, 1.0], [1.0 0.0; 0.0 1.0])),
            (input = (q_μ = MvNormalMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_γ = Gamma(3.0, 2.0)), output = MvNormalMeanPrecision([2.0, 1.0], [6.0 0.0; 0.0 6.0])),
            (
                input = (q_μ = MvNormalWeightedMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_γ = Gamma(4.0, 2.0)),
                output = MvNormalMeanPrecision([3 / 4, -1 / 8], [8.0 0.0; 0.0 8.0])
            )
        ]
    end

    @testset "Structured variational: (m_μ::MultivariateNormalDistributionsFamily, q_γ::Gamma)" begin
        @test_rules [with_float_conversions = true] MvNormalMeanScalePrecision(:out, Marginalisation) [
            (input = (m_μ = MvNormalMeanCovariance([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_γ = Gamma(2.0, 1.0)), output = MvNormalMeanCovariance([2.0, 1.0], [3.5 2.0; 2.0 4.5])),
            (input = (m_μ = MvNormalMeanCovariance([0.0, 1.0], [2.0 -1.0; -1.0 4.0]), q_γ = Gamma(3.0, 1.0)), output = MvNormalMeanCovariance([0.0, 1.0], [7/3 -1.0; -1.0 13/3])),
            (
                input = (m_μ = MvNormalWeightedMeanPrecision([3.0, -1.0], [1.0 0.0; 0.0 1.0]), q_γ = Gamma(4.0, 2.0)),
                output = MvNormalMeanCovariance([3.0, -1.0], [1.125 0.0; 0.0 1.125])
            )
        ]
    end
end

end
