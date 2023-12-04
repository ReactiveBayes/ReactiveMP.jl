module RulesNOTOutTest

using Test, ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

import ReactiveMP: @test_rules

@testset "rules:NOT:out" begin
    @testset "Belief Propagation: (m_in::Bernoulli)" begin
        @test_rules [check_type_promotion = true] NOT(:out, Marginalisation) [
            (input = (m_in = Bernoulli(0.5),), output = Bernoulli(0.5)), (input = (m_in = Bernoulli(0.3),), output = Bernoulli(0.7))
        ]
    end
end
end
