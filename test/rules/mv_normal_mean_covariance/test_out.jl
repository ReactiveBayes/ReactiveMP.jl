module RulesMvNormalMeanCovarianceOutTest

using Test
using ReactiveMP
using Random

import ReactiveMP: @test_rules

@testset "rules:MvNormalMeanCovariance:out" begin
    @testset "Belief Propagation: (m_μ::PointMass, m_Σ::PointMass)" begin
        @test_rules [with_float_conversions = true] MvNormalMeanCovariance(:out, Marginalisation) [
            (input = (m_μ = PointMass([-1.0]), m_Σ = PointMass([2.0])), output = MvNormalMeanCovariance([-1.0], [2.0])),
            (input = (m_μ = PointMass([1.0]), m_Σ = PointMass([2.0])), output = MvNormalMeanCovariance([1.0], [2.0])),
            (input = (m_μ = PointMass([2.0]), m_Σ = PointMass([1.0])), output = MvNormalMeanCovariance([2.0], [1.0])),
            (input = (m_μ = PointMass([1.0, 3.0]), m_Σ = PointMass([3.0 2.0; 2.0 4.0])), output = MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0])),
            (input = (m_μ = PointMass([-1.0, 2.0]), m_Σ = PointMass([7.0 -1.0; -1.0 9.0])), output = MvNormalMeanCovariance([-1.0, 2.0], [7.0 -1.0; -1.0 9.0])),
            (input = (m_μ = PointMass([0.0, 0.0]), m_Σ = PointMass([1.0 0.0; 0.0 1.0])), output = MvNormalMeanCovariance([0.0, 0.0], [1.0 0.0; 0.0 1.0]))
        ]
    end

    @testset "Belief Propagation: (m_μ::MultivariateNormalDistributionsFamily, m_Σ::PointMass)" begin
        @test_rules [with_float_conversions = true] MvNormalMeanCovariance(:out, Marginalisation) [
            (
                input = (m_μ = MvNormalMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), m_Σ = PointMass([6.0 4.0; 4.0 8.0])),
                output = MvNormalMeanCovariance([2.0, 1.0], [13/2 15/4; 15/4 67/8])
            ),
            (
                input = (m_μ = MvNormalMeanPrecision([0.0, 0.0], [7.0 -1.0; -1.0 9.0]), m_Σ = PointMass([12.0 -2.0; -2.0 7.0])),
                output = MvNormalMeanCovariance([0.0, 0.0], [753/62 -123/62; -123/62 441/62])
            ),
            (
                input = (m_μ = MvNormalMeanPrecision([3.0, -1.0], [1.0 0.0; 0.0 1.0]), m_Σ = PointMass([1.0 0.0; 0.0 1.0])),
                output = MvNormalMeanCovariance([3.0, -1.0], [2.0 0.0; 0.0 2.0])
            )
        ]

        @test_rules [with_float_conversions = true] MvNormalMeanCovariance(:out, Marginalisation) [
            (
                input = (m_μ = MvNormalMeanCovariance([2.0, 1.0], [3.0 2.0; 2.0 4.0]), m_Σ = PointMass([6.0 4.0; 4.0 8.0])),
                output = MvNormalMeanCovariance([2.0, 1.0], [9.0 6.0; 6.0 12.0])
            ),
            (
                input = (m_μ = MvNormalMeanCovariance([0.0, 0.0], [7.0 -1.0; -1.0 9.0]), m_Σ = PointMass([12.0 -2.0; -2.0 7.0])),
                output = MvNormalMeanCovariance([0.0, 0.0], [19.0 -3.0; -3.0 16.0])
            ),
            (
                input = (m_μ = MvNormalMeanCovariance([3.0, -1.0], [1.0 0.0; 0.0 1.0]), m_Σ = PointMass([1.0 0.0; 0.0 1.0])),
                output = MvNormalMeanCovariance([3.0, -1.0], [2.0 0.0; 0.0 2.0])
            )
        ]

        @test_rules [with_float_conversions = true] MvNormalMeanCovariance(:out, Marginalisation) [
            (
                input = (m_μ = MvNormalWeightedMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), m_Σ = PointMass([6.0 4.0; 4.0 8.0])),
                output = MvNormalMeanCovariance([3 / 4, -1 / 8], [13/2 15/4; 15/4 67/8])
            ),
            (
                input = (m_μ = MvNormalWeightedMeanPrecision([0.0, 0.0], [7.0 -1.0; -1.0 9.0]), m_Σ = PointMass([12.0 -2.0; -2.0 7.0])),
                output = MvNormalMeanCovariance([0.0, 0.0], [753/62 -123/62; -123/62 441/62])
            ),
            (
                input = (m_μ = MvNormalWeightedMeanPrecision([3.0, -1.0], [1.0 0.0; 0.0 1.0]), m_Σ = PointMass([1.0 0.0; 0.0 1.0])),
                output = MvNormalMeanCovariance([3.0, -1.0], [2.0 0.0; 0.0 2.0])
            )
        ]
    end

    @testset "Variational: (q_μ::Any, q_Σ::Any)" begin
        @test_rules [with_float_conversions = true] MvNormalMeanCovariance(:out, Marginalisation) [
            (input = (q_μ = PointMass([-1.0]), q_Σ = PointMass([2.0])), output = MvNormalMeanCovariance([-1.0], [2.0])),
            (input = (q_μ = PointMass([1.0]), q_Σ = PointMass([2.0])), output = MvNormalMeanCovariance([1.0], [2.0])),
            (input = (q_μ = PointMass([2.0]), q_Σ = PointMass([1.0])), output = MvNormalMeanCovariance([2.0], [1.0])),
            (input = (q_μ = PointMass([1.0, 3.0]), q_Σ = PointMass([3.0 2.0; 2.0 4.0])), output = MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0])),
            (input = (q_μ = PointMass([-1.0, 2.0]), q_Σ = PointMass([7.0 -1.0; -1.0 9.0])), output = MvNormalMeanCovariance([-1.0, 2.0], [7.0 -1.0; -1.0 9.0])),
            (input = (q_μ = PointMass([0.0, 0.0]), q_Σ = PointMass([1.0 0.0; 0.0 1.0])), output = MvNormalMeanCovariance([0.0, 0.0], [1.0 0.0; 0.0 1.0]))
        ]

        @test_rules [with_float_conversions = true] MvNormalMeanCovariance(:out, Marginalisation) [
            (
                input = (q_μ = MvNormalMeanCovariance([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_Σ = PointMass([6.0 4.0; 4.0 8.0])),
                output = MvNormalMeanCovariance([2.0, 1.0], [6.0 4.0; 4.0 8.0])
            ),
            (
                input = (q_μ = MvNormalMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_Σ = PointMass([6.0 4.0; 4.0 8.0])),
                output = MvNormalMeanCovariance([2.0, 1.0], [6.0 4.0; 4.0 8.0])
            ),
            (
                input = (q_μ = MvNormalWeightedMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_Σ = PointMass([6.0 4.0; 4.0 8.0])),
                output = MvNormalMeanCovariance([3 / 4, -1 / 8], [6.0 4.0; 4.0 8.0])
            )
        ]
    end

    @testset "Structured variational: (m_μ::PointMass, q_Σ::Any)" begin
        @test_rules [with_float_conversions = true] MvNormalMeanCovariance(:out, Marginalisation) [
            (input = (m_μ = PointMass([1.0, 3.0]), q_Σ = PointMass([3.0 2.0; 2.0 4.0])), output = MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0])),
            (input = (m_μ = PointMass([-1.0, 2.0]), q_Σ = PointMass([7.0 -1.0; -1.0 9.0])), output = MvNormalMeanCovariance([-1.0, 2.0], [7.0 -1.0; -1.0 9.0])),
            (input = (m_μ = PointMass([0.0, 0.0]), q_Σ = PointMass([1.0 0.0; 0.0 1.0])), output = MvNormalMeanCovariance([0.0, 0.0], [1.0 0.0; 0.0 1.0]))
        ]
    end

    @testset "Structured variational: (m_μ::MultivariateNormalDistributionsFamily, q_Σ::Any)" begin
        @test_rules [with_float_conversions = true] MvNormalMeanCovariance(:out, Marginalisation) [
            (
                input = (m_μ = MvNormalMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_Σ = PointMass([6.0 4.0; 4.0 8.0])),
                output = MvNormalMeanCovariance([2.0, 1.0], [13/2 15/4; 15/4 67/8])
            ),
            (
                input = (m_μ = MvNormalMeanPrecision([0.0, 0.0], [7.0 -1.0; -1.0 9.0]), q_Σ = PointMass([12.0 -2.0; -2.0 7.0])),
                output = MvNormalMeanCovariance([0.0, 0.0], [753/62 -123/62; -123/62 441/62])
            ),
            (
                input = (m_μ = MvNormalMeanPrecision([3.0, -1.0], [1.0 0.0; 0.0 1.0]), q_Σ = PointMass([1.0 0.0; 0.0 1.0])),
                output = MvNormalMeanCovariance([3.0, -1.0], [2.0 0.0; 0.0 2.0])
            )
        ]

        @test_rules [with_float_conversions = true] MvNormalMeanCovariance(:out, Marginalisation) [
            (
                input = (m_μ = MvNormalMeanCovariance([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_Σ = PointMass([6.0 4.0; 4.0 8.0])),
                output = MvNormalMeanCovariance([2.0, 1.0], [9.0 6.0; 6.0 12.0])
            ),
            (
                input = (m_μ = MvNormalMeanCovariance([0.0, 0.0], [7.0 -1.0; -1.0 9.0]), q_Σ = PointMass([12.0 -2.0; -2.0 7.0])),
                output = MvNormalMeanCovariance([0.0, 0.0], [19.0 -3.0; -3.0 16.0])
            ),
            (
                input = (m_μ = MvNormalMeanCovariance([3.0, -1.0], [1.0 0.0; 0.0 1.0]), q_Σ = PointMass([1.0 0.0; 0.0 1.0])),
                output = MvNormalMeanCovariance([3.0, -1.0], [2.0 0.0; 0.0 2.0])
            )
        ]

        @test_rules [with_float_conversions = true] MvNormalMeanCovariance(:out, Marginalisation) [
            (
                input = (m_μ = MvNormalWeightedMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_Σ = PointMass([6.0 4.0; 4.0 8.0])),
                output = MvNormalMeanCovariance([3 / 4, -1 / 8], [13/2 15/4; 15/4 67/8])
            ),
            (
                input = (m_μ = MvNormalWeightedMeanPrecision([0.0, 0.0], [7.0 -1.0; -1.0 9.0]), q_Σ = PointMass([12.0 -2.0; -2.0 7.0])),
                output = MvNormalMeanCovariance([0.0, 0.0], [753/62 -123/62; -123/62 441/62])
            ),
            (
                input = (m_μ = MvNormalWeightedMeanPrecision([3.0, -1.0], [1.0 0.0; 0.0 1.0]), q_Σ = PointMass([1.0 0.0; 0.0 1.0])),
                output = MvNormalMeanCovariance([3.0, -1.0], [2.0 0.0; 0.0 2.0])
            )
        ]
    end
end

end
