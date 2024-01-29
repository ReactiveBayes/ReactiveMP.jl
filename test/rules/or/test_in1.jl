module RulesORIn1Test

using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

import ReactiveMP: @test_rules

@testitem "rules:OR:in1" begin
    @testset "Belief Propagation: (m_out::Bernoulli, m_in2::Bernoulli)" begin
        @test_rules [check_type_promotion = true] OR(:in1, Marginalisation) [
            (input = (m_out = Bernoulli(0.6), m_in2 = Bernoulli(0.5)), output = Bernoulli(0.6 / 1.1)),
            (input = (m_out = Bernoulli(0.3), m_in2 = Bernoulli(0.4)), output = Bernoulli(0.3 / 0.84))
        ]
    end
end
end
