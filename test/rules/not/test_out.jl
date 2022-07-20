module RulesNOTOutTest

using Test
using ReactiveMP
using Random
import ReactiveMP: @test_rules

@testset "rules:typeof(NOT):out" begin
    @testset "Belief Propagation: (m_in::Bernoulli)" begin
        @test_rules [with_float_conversions = true] typeof(NOT)(:out, Marginalisation) [
            (
                input = (m_in = Bernoulli(0.5)),
                output = Bernoulli(0.5)
            ), (
                input = (m_in = Bernoulli(0.3)),
                output = Bernoulli(0.7)
            )
        ]
    end
end
end
