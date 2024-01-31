module RulesMvNormalWeightedMeanPrecisionOutTest

using Test, ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

import ReactiveMP: @test_rules

@testset "rules:MvNormalWeightedMeanPrecision:out" begin
    @testset "Belief Propagation: (m_ξ::PointMass, m_Λ::PointMass)" begin
        @test_rules [check_type_promotion = true] MvNormalWeightedMeanPrecision(:out, Marginalisation) [
            (input = (m_ξ = PointMass([-1.0]), m_Λ = PointMass([2.0])), output = MvNormalWeightedMeanPrecision([-1.0], [2.0])),
            (input = (m_ξ = PointMass([1.0]), m_Λ = PointMass([2.0])), output = MvNormalWeightedMeanPrecision([1.0], [2.0])),
            (input = (m_ξ = PointMass([2.0]), m_Λ = PointMass([1.0])), output = MvNormalWeightedMeanPrecision([2.0], [1.0])),
            (input = (m_ξ = PointMass([1.0, 3.0]), m_Λ = PointMass([3.0 2.0; 2.0 4.0])), output = MvNormalWeightedMeanPrecision([1.0, 3.0], [3.0 2.0; 2.0 4.0])),
            (input = (m_ξ = PointMass([-1.0, 2.0]), m_Λ = PointMass([7.0 -1.0; -1.0 9.0])), output = MvNormalWeightedMeanPrecision([-1.0, 2.0], [7.0 -1.0; -1.0 9.0])),
            (input = (m_ξ = PointMass([0.0, 0.0]), m_Λ = PointMass([1.0 0.0; 0.0 1.0])), output = MvNormalWeightedMeanPrecision([0.0, 0.0], [1.0 0.0; 0.0 1.0]))
        ]
    end

    @testset "Variational: (q_ξ::Any, q_Λ::Any)" begin
        @test_rules [check_type_promotion = true] MvNormalWeightedMeanPrecision(:out, Marginalisation) [
            (input = (q_ξ = PointMass([-1.0]), q_Λ = PointMass([2.0])), output = MvNormalWeightedMeanPrecision([-1.0], [2.0])),
            (input = (q_ξ = PointMass([1.0]), q_Λ = PointMass([2.0])), output = MvNormalWeightedMeanPrecision([1.0], [2.0])),
            (input = (q_ξ = PointMass([2.0]), q_Λ = PointMass([1.0])), output = MvNormalWeightedMeanPrecision([2.0], [1.0])),
            (input = (q_ξ = PointMass([1.0, 3.0]), q_Λ = PointMass([3.0 2.0; 2.0 4.0])), output = MvNormalWeightedMeanPrecision([1.0, 3.0], [3.0 2.0; 2.0 4.0])),
            (input = (q_ξ = PointMass([-1.0, 2.0]), q_Λ = PointMass([7.0 -1.0; -1.0 9.0])), output = MvNormalWeightedMeanPrecision([-1.0, 2.0], [7.0 -1.0; -1.0 9.0])),
            (input = (q_ξ = PointMass([0.0, 0.0]), q_Λ = PointMass([1.0 0.0; 0.0 1.0])), output = MvNormalWeightedMeanPrecision([0.0, 0.0], [1.0 0.0; 0.0 1.0]))
        ]

        @test_rules [check_type_promotion = true] MvNormalWeightedMeanPrecision(:out, Marginalisation) [
            (
                input = (q_ξ = MvNormalWeightedMeanPrecision([3.0 2.0; 2.0 4.0] * [2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_Λ = Wishart(2.0, [6.0 4.0; 4.0 8.0] ./ 2.0)),
                output = MvNormalWeightedMeanPrecision([2.0, 1.0], [6.0 4.0; 4.0 8.0])
            ),
            (
                input = (q_ξ = MvNormalWeightedMeanPrecision([0.0, 0.0], [7.0 -1.0; -1.0 9.0]), q_Λ = Wishart(3.0, [12.0 -2.0; -2.0 7.0] ./ 3.0)),
                output = MvNormalWeightedMeanPrecision([0.0, 0.0], [12.0 -2.0; -2.0 7.0])
            ),
            (
                input = (q_ξ = MvNormalWeightedMeanPrecision([3.0, -1.0], [1.0 0.0; 0.0 1.0]), q_Λ = Wishart(4.0, [1.0 0.0; 0.0 1.0] ./ 4.0)),
                output = MvNormalWeightedMeanPrecision([3.0, -1.0], [1.0 0.0; 0.0 1.0])
            )
        ]
    end
end

end
