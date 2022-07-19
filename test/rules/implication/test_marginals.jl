module RulesImplicationMarginalsTest

using Test
using ReactiveMP
using Random
import ReactiveMP: @test_rules, @test_marginalrules

@testset "rules:IMPL:marginals" begin
    @testset ":in1_in2 (m_out::Bernoulli, m_in1::Bernoulli, m_in2::Bernoulli)" begin
        @test_marginalrules [with_float_conversions = false] IMPL(:in1_in2) [
            (
            input = (
                m_out = Bernoulli(0.5),
                m_in1 = Bernoulli(0.5),
                m_in2 = Bernoulli(0.5)
            ),
            output = (Contingency([0.5^3 0.5^3; 0.5^3 0.5^3])
            )
        )
        ]
    end
end
end
