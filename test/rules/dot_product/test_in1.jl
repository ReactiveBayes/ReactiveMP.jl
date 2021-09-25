module RulesDotProductIn1Test

using Test
using ReactiveMP
using Random
import ReactiveMP: @test_rules
import LinearAlgebra: dot

@testset "rules:typeof(dot):in1" begin

     @testset "Belief Propagation: (m_out::UnivariateNormalDistributionsFamily, m_in2::PointMass)" begin

        @test_rules [ with_float_conversions = true ] typeof(dot)(:in1, Marginalisation) [
            (input = (m_out = NormalMeanVariance(2.0, 2.0), m_in2 = PointMass(-1.0)), output = NormalWeightedMeanPrecision(-1.0, 0.5)),
            (input = (m_out = NormalMeanVariance(1.0, 1.0), m_in2 = PointMass(-2.0)), output = NormalWeightedMeanPrecision(-2.0, 4.0)),
            (input = (m_out = NormalMeanVariance(2.0, 1.0), m_in2 = PointMass(-1.0)), output = NormalWeightedMeanPrecision(-2.0, 1.0)),
        ]

    end

    @testset "Belief Propagation: (m_out::PointMass, m_in2::UnivariateNormalDistributionsFamily)" begin

        @test_rules [ with_float_conversions = true ] typeof(dot)(:in1, Marginalisation) [
            (input = (m_out = PointMass(-1.0), m_in2 = NormalMeanVariance(1.0, 2.0)), output = NormalWeightedMeanPrecision(-0.5, 0.5)),
            (input = (m_out = PointMass(3.0), m_in2 = NormalMeanVariance(2.0, 3.0)), output = NormalWeightedMeanPrecision(2.0, 3.0)),
            (input = (m_out = PointMass(-1.0), m_in2 = NormalMeanVariance(2.0, 1.0)), output = NormalWeightedMeanPrecision(-2.0, 1.0)),
        ]

    end

end
end