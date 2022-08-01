module RulesNOTMarginalsTest

using Test
using ReactiveMP
using Random
import ReactiveMP: @test_rules, @test_marginalrules

@testset "rules:NOT:marginals" begin
    @testset ":in (m_out::Bernoulli, m_in::Bernoulli)" begin
        @test_marginalrules [with_float_conversions = true] NOT(:in) [
            (
                input = (
                    m_out = Bernoulli(0.4),
                    m_in = Bernoulli(0.5)
                ),
                output = Bernoulli(0.6)
            ),
            (
                input = (
                    m_out = Bernoulli(0.2),
                    m_in = Bernoulli(0.8)
                ),
                output = Bernoulli(0.64 / (0.68))
            )
        ]
    end
end
end
