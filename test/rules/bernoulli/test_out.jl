module RulesBernoulliOutTest

using Test
using ReactiveMP
using Random
import ReactiveMP: @test_rules

@testset "rules:Bernoulli:out" begin

    @testset "Belief Propagation: (p::PointMass)" begin

        @test_rules [ with_float_conversions = true ] Bernoulli(:out, Marginalisation) [
            (input = (p = PointMass(1.0)), output = Bernoulli(1.0)),
        ]

    end
end
end