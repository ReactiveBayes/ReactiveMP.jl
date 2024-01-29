module RulesMvNormalMeanCovarianceMeanTest

using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

import ReactiveMP: @test_rules

@testitem "rules:MvNormalMeanCovariance:mean" begin
    @testset "Belief Propagation: (m_out::PointMass, m_Σ::PointMass)" begin
        @test_rules [check_type_promotion = true] MvNormalMeanCovariance(:μ, Marginalisation) [
            (input = (m_out = PointMass([-1.0]), m_Σ = PointMass([2.0])), output = MvNormalMeanCovariance([-1.0], [2.0])),
            (input = (m_out = PointMass([1.0]), m_Σ = PointMass([2.0])), output = MvNormalMeanCovariance([1.0], [2.0])),
            (input = (m_out = PointMass([2.0]), m_Σ = PointMass([1.0])), output = MvNormalMeanCovariance([2.0], [1.0])),
            (input = (m_out = PointMass([1.0, 3.0]), m_Σ = PointMass([3.0 2.0; 2.0 4.0])), output = MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0])),
            (input = (m_out = PointMass([-1.0, 2.0]), m_Σ = PointMass([7.0 -1.0; -1.0 9.0])), output = MvNormalMeanCovariance([-1.0, 2.0], [7.0 -1.0; -1.0 9.0])),
            (input = (m_out = PointMass([0.0, 0.0]), m_Σ = PointMass([1.0 0.0; 0.0 1.0])), output = MvNormalMeanCovariance([0.0, 0.0], [1.0 0.0; 0.0 1.0]))
        ]
    end

    @testset "Belief Propagation: (m_out::MultivariateNormalDistributionsFamily, m_Σ::PointMass)" begin
        @test_rules [check_type_promotion = true] MvNormalMeanCovariance(:μ, Marginalisation) [
            (
                input = (m_out = MvNormalMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), m_Σ = PointMass([6.0 4.0; 4.0 8.0])),
                output = MvNormalMeanCovariance([2.0, 1.0], [13/2 15/4; 15/4 67/8])
            ),
            (
                input = (m_out = MvNormalMeanPrecision([0.0, 0.0], [7.0 -1.0; -1.0 9.0]), m_Σ = PointMass([12.0 -2.0; -2.0 7.0])),
                output = MvNormalMeanCovariance([0.0, 0.0], [753/62 -123/62; -123/62 441/62])
            ),
            (
                input = (m_out = MvNormalMeanPrecision([3.0, -1.0], [1.0 0.0; 0.0 1.0]), m_Σ = PointMass([1.0 0.0; 0.0 1.0])),
                output = MvNormalMeanCovariance([3.0, -1.0], [2.0 0.0; 0.0 2.0])
            )
        ]

        @test_rules [check_type_promotion = true] MvNormalMeanCovariance(:μ, Marginalisation) [
            (
                input = (m_out = MvNormalMeanCovariance([2.0, 1.0], [3.0 2.0; 2.0 4.0]), m_Σ = PointMass([6.0 4.0; 4.0 8.0])),
                output = MvNormalMeanCovariance([2.0, 1.0], [9.0 6.0; 6.0 12.0])
            ),
            (
                input = (m_out = MvNormalMeanCovariance([0.0, 0.0], [7.0 -1.0; -1.0 9.0]), m_Σ = PointMass([12.0 -2.0; -2.0 7.0])),
                output = MvNormalMeanCovariance([0.0, 0.0], [19.0 -3.0; -3.0 16.0])
            ),
            (
                input = (m_out = MvNormalMeanCovariance([3.0, -1.0], [1.0 0.0; 0.0 1.0]), m_Σ = PointMass([1.0 0.0; 0.0 1.0])),
                output = MvNormalMeanCovariance([3.0, -1.0], [2.0 0.0; 0.0 2.0])
            )
        ]

        @test_rules [check_type_promotion = true] MvNormalMeanCovariance(:μ, Marginalisation) [
            (
                input = (m_out = MvNormalWeightedMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), m_Σ = PointMass([6.0 4.0; 4.0 8.0])),
                output = MvNormalMeanCovariance([3 / 4, -1 / 8], [13/2 15/4; 15/4 67/8])
            ),
            (
                input = (m_out = MvNormalWeightedMeanPrecision([0.0, 0.0], [7.0 -1.0; -1.0 9.0]), m_Σ = PointMass([12.0 -2.0; -2.0 7.0])),
                output = MvNormalMeanCovariance([0.0, 0.0], [753/62 -123/62; -123/62 441/62])
            ),
            (
                input = (m_out = MvNormalWeightedMeanPrecision([3.0, -1.0], [1.0 0.0; 0.0 1.0]), m_Σ = PointMass([1.0 0.0; 0.0 1.0])),
                output = MvNormalMeanCovariance([3.0, -1.0], [2.0 0.0; 0.0 2.0])
            )
        ]
    end

    @testset "Variational: (q_out::Any, q_Σ::Any)" begin
        @test_rules [check_type_promotion = true] MvNormalMeanCovariance(:μ, Marginalisation) [
            (input = (q_out = PointMass([-1.0]), q_Σ = PointMass([2.0])), output = MvNormalMeanCovariance([-1.0], [2.0])),
            (input = (q_out = PointMass([1.0]), q_Σ = PointMass([2.0])), output = MvNormalMeanCovariance([1.0], [2.0])),
            (input = (q_out = PointMass([2.0]), q_Σ = PointMass([1.0])), output = MvNormalMeanCovariance([2.0], [1.0])),
            (input = (q_out = PointMass([1.0, 3.0]), q_Σ = PointMass([3.0 2.0; 2.0 4.0])), output = MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0])),
            (input = (q_out = PointMass([-1.0, 2.0]), q_Σ = PointMass([7.0 -1.0; -1.0 9.0])), output = MvNormalMeanCovariance([-1.0, 2.0], [7.0 -1.0; -1.0 9.0])),
            (input = (q_out = PointMass([0.0, 0.0]), q_Σ = PointMass([1.0 0.0; 0.0 1.0])), output = MvNormalMeanCovariance([0.0, 0.0], [1.0 0.0; 0.0 1.0]))
        ]

        @test_rules [check_type_promotion = true] MvNormalMeanCovariance(:μ, Marginalisation) [
            (
                input = (q_out = MvNormalMeanCovariance([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_Σ = PointMass([6.0 4.0; 4.0 8.0])),
                output = MvNormalMeanCovariance([2.0, 1.0], [6.0 4.0; 4.0 8.0])
            ),
            (
                input = (q_out = MvNormalMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_Σ = PointMass([6.0 4.0; 4.0 8.0])),
                output = MvNormalMeanCovariance([2.0, 1.0], [6.0 4.0; 4.0 8.0])
            ),
            (
                input = (q_out = MvNormalWeightedMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_Σ = PointMass([6.0 4.0; 4.0 8.0])),
                output = MvNormalMeanCovariance([3 / 4, -1 / 8], [6.0 4.0; 4.0 8.0])
            )
        ]
    end

    @testset "Structured variational: (m_out::PointMass, q_Σ::Any)" begin
        @test_rules [check_type_promotion = true] MvNormalMeanCovariance(:μ, Marginalisation) [
            (input = (m_out = PointMass([1.0, 3.0]), q_Σ = PointMass([3.0 2.0; 2.0 4.0])), output = MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0])),
            (input = (m_out = PointMass([-1.0, 2.0]), q_Σ = PointMass([7.0 -1.0; -1.0 9.0])), output = MvNormalMeanCovariance([-1.0, 2.0], [7.0 -1.0; -1.0 9.0])),
            (input = (m_out = PointMass([0.0, 0.0]), q_Σ = PointMass([1.0 0.0; 0.0 1.0])), output = MvNormalMeanCovariance([0.0, 0.0], [1.0 0.0; 0.0 1.0]))
        ]
    end

    @testset "Structured variational: (m_out::MultivariateNormalDistributionsFamily, q_Σ::Any)" begin
        @test_rules [check_type_promotion = true] MvNormalMeanCovariance(:μ, Marginalisation) [
            (
                input = (m_out = MvNormalMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_Σ = PointMass([6.0 4.0; 4.0 8.0])),
                output = MvNormalMeanCovariance([2.0, 1.0], [13/2 15/4; 15/4 67/8])
            ),
            (
                input = (m_out = MvNormalMeanPrecision([0.0, 0.0], [7.0 -1.0; -1.0 9.0]), q_Σ = PointMass([12.0 -2.0; -2.0 7.0])),
                output = MvNormalMeanCovariance([0.0, 0.0], [753/62 -123/62; -123/62 441/62])
            ),
            (
                input = (m_out = MvNormalMeanPrecision([3.0, -1.0], [1.0 0.0; 0.0 1.0]), q_Σ = PointMass([1.0 0.0; 0.0 1.0])),
                output = MvNormalMeanCovariance([3.0, -1.0], [2.0 0.0; 0.0 2.0])
            )
        ]

        @test_rules [check_type_promotion = true] MvNormalMeanCovariance(:μ, Marginalisation) [
            (
                input = (m_out = MvNormalMeanCovariance([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_Σ = PointMass([6.0 4.0; 4.0 8.0])),
                output = MvNormalMeanCovariance([2.0, 1.0], [9.0 6.0; 6.0 12.0])
            ),
            (
                input = (m_out = MvNormalMeanCovariance([0.0, 0.0], [7.0 -1.0; -1.0 9.0]), q_Σ = PointMass([12.0 -2.0; -2.0 7.0])),
                output = MvNormalMeanCovariance([0.0, 0.0], [19.0 -3.0; -3.0 16.0])
            ),
            (
                input = (m_out = MvNormalMeanCovariance([3.0, -1.0], [1.0 0.0; 0.0 1.0]), q_Σ = PointMass([1.0 0.0; 0.0 1.0])),
                output = MvNormalMeanCovariance([3.0, -1.0], [2.0 0.0; 0.0 2.0])
            )
        ]

        @test_rules [check_type_promotion = true] MvNormalMeanCovariance(:μ, Marginalisation) [
            (
                input = (m_out = MvNormalWeightedMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_Σ = PointMass([6.0 4.0; 4.0 8.0])),
                output = MvNormalMeanCovariance([3 / 4, -1 / 8], [13/2 15/4; 15/4 67/8])
            ),
            (
                input = (m_out = MvNormalWeightedMeanPrecision([0.0, 0.0], [7.0 -1.0; -1.0 9.0]), q_Σ = PointMass([12.0 -2.0; -2.0 7.0])),
                output = MvNormalMeanCovariance([0.0, 0.0], [753/62 -123/62; -123/62 441/62])
            ),
            (
                input = (m_out = MvNormalWeightedMeanPrecision([3.0, -1.0], [1.0 0.0; 0.0 1.0]), q_Σ = PointMass([1.0 0.0; 0.0 1.0])),
                output = MvNormalMeanCovariance([3.0, -1.0], [2.0 0.0; 0.0 2.0])
            )
        ]
    end
end

end
