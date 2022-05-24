module RulesBernoulliMarginalsTest

using Test
using ReactiveMP
using Random
import ReactiveMP: @test_marginalrules

@testset "marginalrules:Bernoulli" begin
    @testset "out_p: (m_out::PointMass, m_p::Beta)" begin
        @test_marginalrules [with_float_conversions = true] Bernoulli(:out_p) [
            (
                input = (m_out = PointMass(1.0), m_p = Beta(2.0, 1.0)),
                output = (out = PointMass(1.0), p = Beta(3.0, 1.0))
            ),
            (
                input = (m_out = PointMass(1.0), m_p = Beta(4.0, 2.0)),
                output = (out = PointMass(1.0), p = Beta(5.0, 2.0))
            ),
            (
                input = (m_out = PointMass(0.0), m_p = Beta(1.0, 2.0)),
                output = (out = PointMass(0.0), p = Beta(1.0, 3.0))
            )]
    end
end
end
