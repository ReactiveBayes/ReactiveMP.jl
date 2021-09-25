module RulesDotProductMarginalsTest

using Test
using ReactiveMP
using Random
import ReactiveMP: @test_marginalrules
import LinearAlgebra: dot

@testset "marginalrules:DotProduct" begin

     @testset "in1_in2: (m_out::UnivariateNormalDistributionsFamily, m_in1::PointMass, m_in2::UnivariateNormalDistributionsFamily)" begin

        @test_marginalrules [ with_float_conversions = true ] typeof(dot)(:in1_in2) [
            (input = (m_out = NormalMeanVariance(1.0, 2.0), m_in1 = PointMass(1.0), m_in2 = NormalMeanVariance(2.0, 2.0)), output = (in1 = PointMass(1.0), in2 = NormalWeightedMeanPrecision(1.5, 1.0))),
            (input = (m_out = NormalMeanVariance(1.0, 1.0), m_in1 = PointMass(2.0), m_in2 = NormalMeanVariance(1.0, 2.0)), output = (in1 = PointMass(2.0), in2 = NormalWeightedMeanPrecision(2.5, 4.5))),
            (input = (m_out = NormalMeanVariance(1.0, 2.0), m_in1 = PointMass(-2.0), m_in2 = NormalMeanVariance(1.0, 2.0)), output = (in1 = PointMass(-2.0), in2 = NormalWeightedMeanPrecision(-0.5, 2.5))),
        ]

    end

    @testset "in1_in2: (m_out::UnivariateNormalDistributionsFamily, m_in1::UnivariateNormalDistributionsFamily, m_in2::PointMass)" begin

        @test_marginalrules [ with_float_conversions = true ] typeof(dot)(:in1_in2) [
            (input = (m_out = NormalMeanVariance(1.0, 1.0), m_in1 = NormalMeanVariance(1.0, 2.0), m_in2 = PointMass(2.0)), output = (in1 = PointMass(2.0), in2 = NormalWeightedMeanPrecision(2.5, 4.5))),
            (input = (m_out = NormalMeanVariance(3.0, 1.0), m_in1 = NormalMeanVariance(1.0, 2.0), m_in2 = PointMass(2.0)), output = (in1 = PointMass(2.0), in2 = NormalWeightedMeanPrecision(6.5, 4.5))),
            (input = (m_out = NormalMeanVariance(4.0, 1.0), m_in1 = NormalMeanVariance(1.0, 3.0), m_in2 = PointMass(2.0)), output = (in1 = PointMass(2.0), in2 = NormalWeightedMeanPrecision(25/3, 13/3))),
        ]

    end

end
end