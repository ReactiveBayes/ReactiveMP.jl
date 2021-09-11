module RulesBernoulliOutTest

using Test
using ReactiveMP
using Random
import ReactiveMP: @test_rules

@testset "rules:Bernoulli:out" begin

    @testset "Belief Propagation: (m_p::PointMass)" begin

        @test_rules [ with_float_conversions = true ] Bernoulli(:out, Marginalisation) [
            (input = (m_p = PointMass(1.0),), output = Bernoulli(1.0)),
            (input = (m_p = PointMass(0.2),), output = Bernoulli(0.2)),
        ]

    end

    @testset "Belief Propagation: (q_p::PointMass)" begin

        @test_rules [ with_float_conversions = true ] Bernoulli(:out, Marginalisation) [
            (input = (q_p = PointMass(1.0),), output = Bernoulli(1.0)),
            (input = (q_p = PointMass(0.3),), output = Bernoulli(0.3)),
        ]

    end

    @testset "Belief Propagation: (q_p::Beta)" begin

        @test_rules [ with_float_conversions = true ] Bernoulli(:out, Marginalisation) [
            (input = (q_p = Beta(1.0, 1.0),), output = Bernoulli(0.5)),
            (input = (q_p = Beta(0.2, 0.2),), output = Bernoulli(0.5)),
        ]

    end
end
end