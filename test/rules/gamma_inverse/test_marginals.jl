module RulesGammaInverseMarginalsTest

using Test
using ReactiveMP
using Random
using Distributions

import ReactiveMP: @test_marginalrules

# TODO
@testset "marginalrules:GammaInverse" begin
    @testset "out_α_θ: (m_out::GammaInverse, m_α::PointMass, m_θ::PointMass)" begin
        @test_marginalrules [with_float_conversions = false] GammaInverse(:out_α_θ) [
            (
                input = (m_out = GammaInverse(1.0, 2.0), m_α = PointMass(1.0), m_θ = PointMass(2.0)),
                output = (out = GammaInverse(3.0, 4.0), α = PointMass(1.0), θ = PointMass(2.0))
            )
        ]
    end
end
end
