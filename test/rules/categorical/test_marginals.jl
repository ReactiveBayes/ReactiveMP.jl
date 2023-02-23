module RulesCategoricalMarginalsTest

using Test
using ReactiveMP
using Random
using LinearAlgebra
import ReactiveMP: @test_marginalrules

@testset "marginalrules:Categorical" begin
    @testset "out_p: (m_out::PointMass, m_p::Dirichlet)" begin
        @test_marginalrules [with_float_conversions = true] Categorical(:out_p) [
            (input = (m_out = PointMass([0.0, 1.0]), m_p = Dirichlet([2.0, 1.0])), output = (out = PointMass([0.0, 1.0]), p = Dirichlet([2.0, 2.0]))),
            (input = (m_out = PointMass([0.0, 1.0]), m_p = Dirichlet([4.0, 2.0])), output = (out = PointMass([0.0, 1.0]), p = Dirichlet([4.0, 3.0]))),
            (input = (m_out = PointMass([1.0, 0.0]), m_p = Dirichlet([1.0, 2.0])), output = (out = PointMass([1.0, 0.0]), p = Dirichlet([2.0, 2.0])))
        ]
    end
    @testset "out_p: (m_out::Categorical, m_p::PointMass)" begin
        @test_marginalrules [with_float_conversions = false] Categorical(:out_p) [
            (input = (m_out = Categorical([0.2, 0.8]), m_p = PointMass([0.0, 1.0])), output = (out = Categorical(normalize([tiny, 0.8], 1)), p = PointMass([0.0, 1.0]))),
            (input = (m_out = Categorical([0.8, 0.2]), m_p = PointMass([0.0, 1.0])), output = (out = Categorical(normalize([tiny, 0.2], 1)), p = PointMass([0.0, 1.0]))),
            (input = (m_out = Categorical([0.8, 0.2]), m_p = PointMass([1.0, 0.0])), output = (out = Categorical(normalize([0.8, tiny], 1)), p = PointMass([1.0, 0.0])))
        ]
    end
end
end
