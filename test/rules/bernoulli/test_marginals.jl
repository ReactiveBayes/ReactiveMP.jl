module RulesBernoulliMarginalsTest

using Test, ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

import ReactiveMP: @test_marginalrules

@testset "marginalrules:Bernoulli" begin
    @testset "out_p: (m_out::PointMass, m_p::Beta)" begin
        @test_marginalrules [check_type_promotion = true] Bernoulli(:out_p) [
            (input = (m_out = PointMass(1.0), m_p = Beta(2.0, 1.0)), output = (out = PointMass(1.0), p = Beta(3.0, 1.0))),
            (input = (m_out = PointMass(1.0), m_p = Beta(4.0, 2.0)), output = (out = PointMass(1.0), p = Beta(5.0, 2.0))),
            (input = (m_out = PointMass(0.0), m_p = Beta(1.0, 2.0)), output = (out = PointMass(0.0), p = Beta(1.0, 3.0)))
        ]
    end
    @testset "out_p: (m_out::Bernoulli, m_p::PointMass)" begin
        @test_marginalrules [check_type_promotion = true] Bernoulli(:out_p) [
            (input = (m_out = Bernoulli(0.8), m_p = PointMass(1.0)), output = (out = Bernoulli(1.0), p = PointMass(1.0))),
            (input = (m_out = Bernoulli(0.2), m_p = PointMass(1.0)), output = (out = Bernoulli(1.0), p = PointMass(1.0))),
            (input = (m_out = Bernoulli(0.2), m_p = PointMass(0.0)), output = (out = Bernoulli(0.0), p = PointMass(0.0)))
        ]
    end
end
end
