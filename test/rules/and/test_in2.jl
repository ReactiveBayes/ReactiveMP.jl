module RulesANDIn2Test

using Test
using ReactiveMP
using Random
import ReactiveMP: @test_rules

@testset "rules:AND:in2" begin
    @testset "Belief Propagation: (m_out::Bernoulli, m_in1::Bernoulli)" begin
        @test_rules [with_float_conversions = true] AND(:in2, Marginalisation) [
            (input = (m_out = Bernoulli(0.6), m_in1 = Bernoulli(0.5)), output = Bernoulli(0.5 / 0.9)),
            (input = (m_out = Bernoulli(0.3), m_in1 = Bernoulli(0.4)), output = Bernoulli(0.54 / 1.24))
        ]
    end
end
end
