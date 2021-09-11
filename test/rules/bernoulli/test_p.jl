module RulesBernoulliPTest

using Test
using ReactiveMP
using Random
import ReactiveMP: @test_rules

@testset "rules:Bernoulli:p" begin

    @testset "Belief Propagation: (m_out::PointMass)" begin

        @test_rules [ with_float_conversions = true ] Bernoulli(:p, Marginalisation) [
            (input = (m_out = PointMass(1.0),), output = Beta(2.0, 1.0)),
            (input = (m_out = PointMass(0.2),), output = Beta(12/10, 9/5)),
        ]

    end

    @testset "Belief Propagation: (q_out::PointMass)" begin

        @test_rules [ with_float_conversions = true ] Bernoulli(:p, Marginalisation) [
            (input = (q_out = Bernoulli(1.0),), output = Beta(2.0, 1.0)),
            (input = (q_out = Bernoulli(0.3),), output = Beta(13/10, 17/10)),
        ]

    end
end
end