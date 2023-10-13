module RulesDirichletMarginalsTest

using Test, ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

import ReactiveMP: @test_marginalrules

@testset "marginalrules:Dirichlet" begin
    @testset "out_a: (m_out::Dirichlet, m_a::PointMass)" begin
        @test_marginalrules [check_type_promotion = true] Dirichlet(:out_a) [
            (input = (m_out = Dirichlet([1.0, 2.0]), m_a = PointMass([0.2, 1.0])), output = (out = Dirichlet([0.2, 2.0]), a = PointMass([0.2, 1.0]))),
            (input = (m_out = Dirichlet([2.0, 2.0]), m_a = PointMass([2.0, 0.5])), output = (out = Dirichlet([3.0, 1.5]), a = PointMass([2.0, 0.5]))),
            (input = (m_out = Dirichlet([2.0, 3.0]), m_a = PointMass([3.0, 1.0])), output = (out = Dirichlet([4.0, 3.0]), a = PointMass([3.0, 1.0])))
        ]
    end
end
end
