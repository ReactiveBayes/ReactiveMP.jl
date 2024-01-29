module RulesANDIn1Test

using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

import ReactiveMP: @test_rules

@testitem "rules:AND:in1" begin
    @testset "Belief Propagation: (m_out::Bernoulli, m_in2::Bernoulli)" begin
        @test_rules [check_type_promotion = true] AND(:in1, Marginalisation) [
            (input = (m_out = Bernoulli(0.6), m_in2 = Bernoulli(0.5)), output = Bernoulli(0.5 / 0.9)),
            (input = (m_out = Bernoulli(0.3), m_in2 = Bernoulli(0.4)), output = Bernoulli(0.54 / 1.24))
        ]
    end
end
end
