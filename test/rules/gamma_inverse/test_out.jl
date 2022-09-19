module RulesGammaInverseOutTest

using Test
using ReactiveMP
using Random
using Distributions

import ReactiveMP: @test_rules

@testset "rules:GammaInverse:out" begin
    @testset "Belief Propagation: (m_α::PointMass, m_β::PointMass)" begin
        @test_rules [with_float_conversions = true] GammaInverse(:out, Marginalisation) [
            # TODO
        ]
    end

    @testset "Variational: (m_α::PointMass, q_β::Any)" begin
        @test_rules [with_float_conversions = true] GammaInverse(:out, Marginalisation) [
            # TODO
        ]
    end

    @testset "Variational: (q_α::PointMass, m_β::Any)" begin
        @test_rules [with_float_conversions = true] GammaInverse(:out, Marginalisation) [
            # TODO
        ]
    end

    @testset "Variational: (q_α::PointMass, q_β::Any)" begin
        @test_rules [with_float_conversions = true] GammaInverse(:out, Marginalisation) [
            # TODO
        ]
    end

end # testset
end # module
