module RulesBetaMarginalsTest

using Test
using ReactiveMP
using Random
import ReactiveMP: @test_marginalrules

@testset "marginalrules:Beta" begin

     @testset "m_out, m_a, m_b: (m_out::Beta, m_a::PointMass, m_b::PointMass)" begin

        @test_marginalrules [ with_float_conversions = false ] Beta(:out_a_b) [
            (input = (m_out = Beta(1.0, 2.0), m_a = PointMass(1.0), m_b = PointMass(2.0)), output = (out = Beta(1.0, 3.0), a = PointMass(1.0), b = PointMass(2.0))),
            (input = (m_out = Beta(2.0, 2.0), m_a = PointMass(2.0), m_b = PointMass(3.0)), output = (out = Beta(3.0, 4.0), a = PointMass(2.0), b = PointMass(3.0))),
            (input = (m_out = Beta(2.0, 3.0), m_a = PointMass(1.0), m_b = PointMass(3.0)), output = (out = Beta(2.0, 5.0), a = PointMass(1.0), b = PointMass(3.0))),
        ]

    end
end
end