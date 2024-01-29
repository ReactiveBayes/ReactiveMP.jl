module RulesMvNormalMeanPrecisionOutTest

using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

import ReactiveMP: @test_rules

@testitem "rules:MvNormalMeanPrecision:out" begin
    @testset "Belief Propagation: (m_μ::PointMass, m_Λ::PointMass)" begin
        @test_rules [check_type_promotion = true] MvNormalMeanPrecision(:out, Marginalisation) [
            (input = (m_μ = PointMass([-1.0]), m_Λ = PointMass([2.0])), output = MvNormalMeanPrecision([-1.0], [2.0])),
            (input = (m_μ = PointMass([1.0]), m_Λ = PointMass([2.0])), output = MvNormalMeanPrecision([1.0], [2.0])),
            (input = (m_μ = PointMass([2.0]), m_Λ = PointMass([1.0])), output = MvNormalMeanPrecision([2.0], [1.0])),
            (input = (m_μ = PointMass([1.0, 3.0]), m_Λ = PointMass([3.0 2.0; 2.0 4.0])), output = MvNormalMeanPrecision([1.0, 3.0], [3.0 2.0; 2.0 4.0])),
            (input = (m_μ = PointMass([-1.0, 2.0]), m_Λ = PointMass([7.0 -1.0; -1.0 9.0])), output = MvNormalMeanPrecision([-1.0, 2.0], [7.0 -1.0; -1.0 9.0])),
            (input = (m_μ = PointMass([0.0, 0.0]), m_Λ = PointMass([1.0 0.0; 0.0 1.0])), output = MvNormalMeanPrecision([0.0, 0.0], [1.0 0.0; 0.0 1.0]))
        ]
    end

    @testset "Belief Propagation: (m_μ::MultivariateNormalDistributionsFamily, m_Λ::PointMass)" begin
        @test_rules [check_type_promotion = true] MvNormalMeanPrecision(:out, Marginalisation) [
            (
                input = (m_μ = MvNormalMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), m_Λ = PointMass([6.0 4.0; 4.0 8.0])),
                output = MvNormalMeanCovariance([2.0, 1.0], [0.75 -0.375; -0.375 0.5625])
            ),
            (
                input = (m_μ = MvNormalMeanPrecision([0.0, 0.0], [7.0 -1.0; -1.0 9.0]), m_Λ = PointMass([12.0 -2.0; -2.0 7.0])),
                output = MvNormalMeanCovariance([0.0, 0.0], [577/2480 51/1240; 51/1240 163/620])
            ),
            (
                input = (m_μ = MvNormalMeanPrecision([3.0, -1.0], [1.0 0.0; 0.0 1.0]), m_Λ = PointMass([1.0 0.0; 0.0 1.0])),
                output = MvNormalMeanCovariance([3.0, -1.0], [2.0 0.0; 0.0 2.0])
            )
        ]

        @test_rules [check_type_promotion = true] MvNormalMeanPrecision(:out, Marginalisation) [
            (
                input = (m_μ = MvNormalMeanCovariance([2.0, 1.0], [3.0 2.0; 2.0 4.0]), m_Λ = PointMass([6.0 4.0; 4.0 8.0])),
                output = MvNormalMeanCovariance([2.0, 1.0], [13/4 15/8; 15/8 67/16])
            ),
            (
                input = (m_μ = MvNormalMeanCovariance([0.0, 0.0], [7.0 -1.0; -1.0 9.0]), m_Λ = PointMass([12.0 -2.0; -2.0 7.0])),
                output = MvNormalMeanCovariance([0.0, 0.0], [567/80 -39/40; -39/40 183/20])
            ),
            (
                input = (m_μ = MvNormalMeanCovariance([3.0, -1.0], [1.0 0.0; 0.0 1.0]), m_Λ = PointMass([1.0 0.0; 0.0 1.0])),
                output = MvNormalMeanCovariance([3.0, -1.0], [2.0 0.0; 0.0 2.0])
            )
        ]

        @test_rules [check_type_promotion = true] MvNormalMeanPrecision(:out, Marginalisation) [
            (
                input = (m_μ = MvNormalWeightedMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), m_Λ = PointMass([6.0 4.0; 4.0 8.0])),
                output = MvNormalMeanCovariance([3 / 4, -1 / 8], [0.75 -0.375; -0.375 0.5625])
            ),
            (
                input = (m_μ = MvNormalWeightedMeanPrecision([0.0, 0.0], [7.0 -1.0; -1.0 9.0]), m_Λ = PointMass([12.0 -2.0; -2.0 7.0])),
                output = MvNormalMeanCovariance([0.0, 0.0], [577/2480 51/1240; 51/1240 163/620])
            ),
            (
                input = (m_μ = MvNormalWeightedMeanPrecision([3.0, -1.0], [1.0 0.0; 0.0 1.0]), m_Λ = PointMass([1.0 0.0; 0.0 1.0])),
                output = MvNormalMeanCovariance([3.0, -1.0], [2.0 0.0; 0.0 2.0])
            )
        ]
    end

    @testset "Variational: (q_out::Any, q_Λ::Any)" begin
        @test_rules [check_type_promotion = true] MvNormalMeanPrecision(:out, Marginalisation) [
            (input = (q_μ = PointMass([-1.0]), q_Λ = PointMass([2.0])), output = MvNormalMeanPrecision([-1.0], [2.0])),
            (input = (q_μ = PointMass([1.0]), q_Λ = PointMass([2.0])), output = MvNormalMeanPrecision([1.0], [2.0])),
            (input = (q_μ = PointMass([2.0]), q_Λ = PointMass([1.0])), output = MvNormalMeanPrecision([2.0], [1.0])),
            (input = (q_μ = PointMass([1.0, 3.0]), q_Λ = PointMass([3.0 2.0; 2.0 4.0])), output = MvNormalMeanPrecision([1.0, 3.0], [3.0 2.0; 2.0 4.0])),
            (input = (q_μ = PointMass([-1.0, 2.0]), q_Λ = PointMass([7.0 -1.0; -1.0 9.0])), output = MvNormalMeanPrecision([-1.0, 2.0], [7.0 -1.0; -1.0 9.0])),
            (input = (q_μ = PointMass([0.0, 0.0]), q_Λ = PointMass([1.0 0.0; 0.0 1.0])), output = MvNormalMeanPrecision([0.0, 0.0], [1.0 0.0; 0.0 1.0]))
        ]

        @test_rules [check_type_promotion = true] MvNormalMeanPrecision(:out, Marginalisation) [
            (
                input = (q_μ = MvNormalMeanCovariance([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_Λ = PointMass([6.0 4.0; 4.0 8.0])),
                output = MvNormalMeanPrecision([2.0, 1.0], [6.0 4.0; 4.0 8.0])
            ),
            (
                input = (q_μ = MvNormalMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_Λ = PointMass([6.0 4.0; 4.0 8.0])),
                output = MvNormalMeanPrecision([2.0, 1.0], [6.0 4.0; 4.0 8.0])
            ),
            (
                input = (q_μ = MvNormalWeightedMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_Λ = PointMass([6.0 4.0; 4.0 8.0])),
                output = MvNormalMeanPrecision([3 / 4, -1 / 8], [6.0 4.0; 4.0 8.0])
            )
        ]
    end

    @testset "Structured variational: (m_μ::PointMass, q_Σ::Any)" begin
        @test_rules [check_type_promotion = true] MvNormalMeanPrecision(:out, Marginalisation) [
            (input = (q_μ = PointMass([1.0, 3.0]), q_Λ = PointMass([3.0 2.0; 2.0 4.0])), output = MvNormalMeanPrecision([1.0, 3.0], [3.0 2.0; 2.0 4.0])),
            (input = (q_μ = PointMass([-1.0, 2.0]), q_Λ = PointMass([7.0 -1.0; -1.0 9.0])), output = MvNormalMeanPrecision([-1.0, 2.0], [7.0 -1.0; -1.0 9.0])),
            (input = (q_μ = PointMass([0.0, 0.0]), q_Λ = PointMass([1.0 0.0; 0.0 1.0])), output = MvNormalMeanPrecision([0.0, 0.0], [1.0 0.0; 0.0 1.0]))
        ]
    end

    @testset "Structured variational: (m_μ::MvNormalMeanPrecision, q_Λ::Any)" begin
        @test_rules [check_type_promotion = true] MvNormalMeanPrecision(:out, Marginalisation) [
            (
                input = (m_μ = MvNormalMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_Λ = PointMass([6.0 4.0; 4.0 8.0])),
                output = MvNormalMeanCovariance([2.0, 1.0], [0.75 -0.375; -0.375 0.5625])
            ),
            (
                input = (m_μ = MvNormalMeanPrecision([0.0, 0.0], [7.0 -1.0; -1.0 9.0]), q_Λ = PointMass([12.0 -2.0; -2.0 7.0])),
                output = MvNormalMeanCovariance([0.0, 0.0], [577/2480 51/1240; 51/1240 163/620])
            ),
            (
                input = (m_μ = MvNormalMeanPrecision([3.0, -1.0], [1.0 0.0; 0.0 1.0]), q_Λ = PointMass([1.0 0.0; 0.0 1.0])),
                output = MvNormalMeanCovariance([3.0, -1.0], [2.0 0.0; 0.0 2.0])
            )
        ]

        @test_rules [check_type_promotion = true] MvNormalMeanPrecision(:out, Marginalisation) [
            (
                input = (m_μ = MvNormalMeanCovariance([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_Λ = PointMass([6.0 4.0; 4.0 8.0])),
                output = MvNormalMeanCovariance([2.0, 1.0], [13/4 15/8; 15/8 67/16])
            ),
            (
                input = (m_μ = MvNormalMeanCovariance([0.0, 0.0], [7.0 -1.0; -1.0 9.0]), q_Λ = PointMass([12.0 -2.0; -2.0 7.0])),
                output = MvNormalMeanCovariance([0.0, 0.0], [567/80 -39/40; -39/40 183/20])
            ),
            (
                input = (m_μ = MvNormalMeanCovariance([3.0, -1.0], [1.0 0.0; 0.0 1.0]), q_Λ = PointMass([1.0 0.0; 0.0 1.0])),
                output = MvNormalMeanCovariance([3.0, -1.0], [2.0 0.0; 0.0 2.0])
            )
        ]

        @test_rules [check_type_promotion = true] MvNormalMeanPrecision(:out, Marginalisation) [
            (
                input = (m_μ = MvNormalWeightedMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_Λ = PointMass([6.0 4.0; 4.0 8.0])),
                output = MvNormalMeanCovariance([3 / 4, -1 / 8], [0.75 -0.375; -0.375 0.5625])
            ),
            (
                input = (m_μ = MvNormalWeightedMeanPrecision([0.0, 0.0], [7.0 -1.0; -1.0 9.0]), q_Λ = PointMass([12.0 -2.0; -2.0 7.0])),
                output = MvNormalMeanCovariance([0.0, 0.0], [577/2480 51/1240; 51/1240 163/620])
            ),
            (
                input = (m_μ = MvNormalWeightedMeanPrecision([3.0, -1.0], [1.0 0.0; 0.0 1.0]), q_Λ = PointMass([1.0 0.0; 0.0 1.0])),
                output = MvNormalMeanCovariance([3.0, -1.0], [2.0 0.0; 0.0 2.0])
            )
        ]

        @test_rules [check_type_promotion = true] MvNormalMeanPrecision(:out, Marginalisation) [
            (
                input = (m_μ = MvNormalMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_Λ = Wishart(2.0, [6.0 4.0; 4.0 8.0] ./ 2.0)),
                output = MvNormalMeanCovariance([2.0, 1.0], [0.75 -0.375; -0.375 0.5625])
            ),
            (
                input = (m_μ = MvNormalMeanPrecision([0.0, 0.0], [7.0 -1.0; -1.0 9.0]), q_Λ = Wishart(3.0, [12.0 -2.0; -2.0 7.0] ./ 3.0)),
                output = MvNormalMeanCovariance([0.0, 0.0], [577/2480 51/1240; 51/1240 163/620])
            ),
            (
                input = (m_μ = MvNormalMeanPrecision([3.0, -1.0], [1.0 0.0; 0.0 1.0]), q_Λ = Wishart(4.0, [1.0 0.0; 0.0 1.0] ./ 4.0)),
                output = MvNormalMeanCovariance([3.0, -1.0], [2.0 0.0; 0.0 2.0])
            )
        ]

        @test_rules [check_type_promotion = true] MvNormalMeanPrecision(:out, Marginalisation) [
            (
                input = (m_μ = MvNormalMeanCovariance([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_Λ = Wishart(2.0, [6.0 4.0; 4.0 8.0] ./ 2.0)),
                output = MvNormalMeanCovariance([2.0, 1.0], [13/4 15/8; 15/8 67/16])
            ),
            (
                input = (m_μ = MvNormalMeanCovariance([0.0, 0.0], [7.0 -1.0; -1.0 9.0]), q_Λ = Wishart(3.0, [12.0 -2.0; -2.0 7.0] ./ 3.0)),
                output = MvNormalMeanCovariance([0.0, 0.0], [567/80 -39/40; -39/40 183/20])
            ),
            (
                input = (m_μ = MvNormalMeanCovariance([3.0, -1.0], [1.0 0.0; 0.0 1.0]), q_Λ = Wishart(4.0, [1.0 0.0; 0.0 1.0] ./ 4.0)),
                output = MvNormalMeanCovariance([3.0, -1.0], [2.0 0.0; 0.0 2.0])
            )
        ]

        @test_rules [check_type_promotion = true] MvNormalMeanPrecision(:out, Marginalisation) [
            (
                input = (m_μ = MvNormalWeightedMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_Λ = Wishart(2.0, [6.0 4.0; 4.0 8.0] ./ 2.0)),
                output = MvNormalMeanCovariance([3 / 4, -1 / 8], [0.75 -0.375; -0.375 0.5625])
            ),
            (
                input = (m_μ = MvNormalWeightedMeanPrecision([0.0, 0.0], [7.0 -1.0; -1.0 9.0]), q_Λ = Wishart(3.0, [12.0 -2.0; -2.0 7.0] ./ 3.0)),
                output = MvNormalMeanCovariance([0.0, 0.0], [577/2480 51/1240; 51/1240 163/620])
            ),
            (
                input = (m_μ = MvNormalWeightedMeanPrecision([3.0, -1.0], [1.0 0.0; 0.0 1.0]), q_Λ = Wishart(4.0, [1.0 0.0; 0.0 1.0] ./ 4.0)),
                output = MvNormalMeanCovariance([3.0, -1.0], [2.0 0.0; 0.0 2.0])
            )
        ]
    end
end

end
