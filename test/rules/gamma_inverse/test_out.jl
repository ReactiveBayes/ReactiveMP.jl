module RulesGammaInverseOutTest

using Test
using ReactiveMP
using Random
using Distributions

import ReactiveMP: @test_rules

@testset "rules:GammaInverse:out" begin
    @testset "Belief Propagation: (m_α::Any, m_θ::Any)" begin
        @test_rules [with_float_conversions = true] GammaInverse(:out, Marginalisation) [
            (
                input = (
                    m_α = PointMass(1.0),
                    m_θ = PointMass(2.0)
                ),
                output = GammaInverse(1.0, 2.0)
            )
        ]
    end

    @testset "Variational Message Passing: (q_α::Any, q_θ::Any)" begin
        @test_rules [with_float_conversions = true] GammaInverse(:out, Marginalisation) [
            (
                input = (
                    q_α = PointMass(1.0),
                    q_θ = PointMass(2.0)
                ),
                output = GammaInverse(1.0, 2.0)
            ),
            (
                input = (
                    q_α = Gamma(1.0, 1.0),
                    q_θ = Beta(1.0, 1.0)
                ),
                output = GammaInverse(1.0, 0.5)
            )
        ]
    end
end
end
