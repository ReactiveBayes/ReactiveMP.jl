module RulesANDIn2Test

using Test
using ReactiveMP
using Random
import ReactiveMP: @test_rules

@testset "rules:IMPL:in2" begin
    @testset "Belief Propagation: (m_out::Bernoulli, m_in1::Bernoulli)" begin
        @test_rules [with_float_conversions = true] IMPL(:in2, Marginalisation) [
            (
            input = (m_out = Bernoulli(0.6), m_in1 = Bernoulli(0.5)),
            output = Bernoulli(0.6 / 1.1)
        )
        ]
    end
end
end
