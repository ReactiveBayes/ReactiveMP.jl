module RulesBetaOutTest

using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

import ReactiveMP: @test_rules

@testitem "rules:Beta:out" begin
    @testset "Belief Propagation: (m_a::PointMass, m_b::PointMass)" begin
        @test_rules [check_type_promotion = true] Beta(:out, Marginalisation) [
            (input = (m_a = PointMass(1.0), m_b = PointMass(2.0)), output = Beta(1.0, 2.0)),
            (input = (m_a = PointMass(2.0), m_b = PointMass(2.0)), output = Beta(2.0, 2.0)),
            (input = (m_a = PointMass(3.0), m_b = PointMass(3.0)), output = Beta(3.0, 3.0))
        ]
    end

    @testset "Variational Message Passing: (q_a::PointMass, q_b::PointMass)" begin
        @test_rules [check_type_promotion = true] Beta(:out, Marginalisation) [
            (input = (q_a = PointMass(1.0), q_b = PointMass(2.0)), output = Beta(1.0, 2.0)),
            (input = (q_a = PointMass(2.0), q_b = PointMass(2.0)), output = Beta(2.0, 2.0)),
            (input = (q_a = PointMass(3.0), q_b = PointMass(3.0)), output = Beta(3.0, 3.0))
        ]
    end
end
end
