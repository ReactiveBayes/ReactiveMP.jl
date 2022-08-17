module RulesORMarginalsTest

using Test
using ReactiveMP
using Random
import ReactiveMP: @test_rules, @test_marginalrules

@testset "rules:OR:marginals" begin
    @testset ":in1_in2 (m_out::Bernoulli, m_in1::Bernoulli, m_in2::Bernoulli)" begin
        @test_marginalrules [with_float_conversions = true] OR(:in1_in2) [
            (
                input = (
                    m_out = Bernoulli(0.5),
                    m_in1 = Bernoulli(0.5),
                    m_in2 = Bernoulli(0.5)
                ),
                output = (Contingency([0.5^3 0.5^3; 0.5^3 0.5^3])
                )
            ),
            (
                input = (
                    m_out = Bernoulli(0.2),
                    m_in1 = Bernoulli(0.8),
                    m_in2 = Bernoulli(0.4)
                ),
                output = (Contingency([0.8*0.2*0.6 0.2*0.4*0.2; 0.8*0.2*0.6 0.2*0.8*0.4])
                )
            )
        ]
    end
end
end
