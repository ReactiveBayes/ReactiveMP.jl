module RulesImplicationOutTest

using Test
using ReactiveMP
using Random
import ReactiveMP: @test_rules

@testset "rules:IMPL:out" begin
    @testset "Belief Propagation: (m_in1::Bernoulli, m_in2::Bernoulli)" begin
        @test_rules [with_float_conversions = true] IMPL(:out, Marginalisation) [
            (
                input = (m_in1 = Bernoulli(0.3), m_in2 = Bernoulli(0.5)),
                output = Bernoulli(0.85)
            ),
        ]
    end
end
end