module RulesGammaInverseMarginalsTest

using Test
using ReactiveMP
using Random
using Distributions

import ReactiveMP: @test_marginalrules

@testset "marginalrules:GammaInverse" begin
    @testset ":out_α_β (m_out::GammaInverse, m_α::PointMass, m_β::PointMass)" begin
        @test_marginalrules [with_float_conversions = true, atol = 1e-5] GammaInverse(:out_α_β) [
            (
                input = (
                    m_out = GammaInverse(1.0, 2.0),
                    m_α = PointMass(1.0),
                    m_β = PointMass(2.0)
                ),
                # TODO
                output = (
                    m_out = GammaInverse(1.0, 2.0),
                    m_α = PointMass(1.0),
                    m_β = PointMass(2.0)
                )
            )
        ]
    end
end # testset
end # module
