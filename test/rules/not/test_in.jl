module RulesNOTInTest

using Test
using ReactiveMP
using Random
import ReactiveMP: @test_rules

@testset "rules:NOT:in" begin
    @testset "Belief Propagation: (m_in::Bernoulli)" begin
        @test_rules [with_float_conversions = true] NOT(:in, Marginalisation) [
            (
                input = (m_out = Bernoulli(0.6),),
                output = Bernoulli(0.4)
            ),
            (
                input = (m_out = Bernoulli(0.3),),
                output = Bernoulli(0.7)
            )
        ]
    end
end
end
