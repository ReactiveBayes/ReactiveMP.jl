module RulesDotProductMarginalsTest

using Test
using ReactiveMP
using Random

import ReactiveMP: @test_marginalrules
import LinearAlgebra: dot

@testset "marginalrules:DotProduct" begin

     @testset "in1_in2: (m_out::UnivariateNormalDistributionsFamily, m_in1::PointMass, m_in2::UnivariateNormalDistributionsFamily)" begin

        @test_marginalrules [ with_float_conversions = true ] typeof(dot)(:in1_in2) [
            (input = (m_out = NormalMeanVariance(1.0, 2.0), m_in1 = PointMass(1.0), m_in2 = NormalMeanVariance(2.0, 2.0), meta = NoCorrection()), output = (in1 = PointMass(1.0), in2 = NormalWeightedMeanPrecision(1.5, 1.0))),
            (input = (m_out = NormalMeanPrecision(1.0, 1.0), m_in1 = PointMass(2.0), m_in2 = NormalMeanVariance(1.0, 2.0), meta = NoCorrection()), output = (in1 = PointMass(2.0), in2 = NormalWeightedMeanPrecision(2.5, 4.5))),
            (input = (m_out = NormalWeightedMeanPrecision(0.5, 0.5), m_in1 = PointMass(-2.0), m_in2 = NormalMeanVariance(1.0, 2.0), meta = NoCorrection()), output = (in1 = PointMass(-2.0), in2 = NormalWeightedMeanPrecision(-0.5, 2.5))),
        ]

        @test_marginalrules [ with_float_conversions = false ] typeof(dot)(:in1_in2) [
            (input = (m_out = NormalMeanVariance(1.0, 2.0), m_in1 = PointMass(1.0), m_in2 = NormalMeanVariance(2.0, 2.0), meta = TinyCorrection()), output = (in1 = PointMass(1.0), in2 = NormalWeightedMeanPrecision(1.5, 1.0))),
            (input = (m_out = NormalMeanPrecision(1.0, 1.0), m_in1 = PointMass(2.0), m_in2 = NormalMeanVariance(1.0, 2.0), meta = TinyCorrection()), output = (in1 = PointMass(2.0), in2 = NormalWeightedMeanPrecision(2.5, 4.5))),
            (input = (m_out = NormalWeightedMeanPrecision(0.5, 0.5), m_in1 = PointMass(-2.0), m_in2 = NormalMeanVariance(1.0, 2.0), meta = TinyCorrection()), output = (in1 = PointMass(-2.0), in2 = NormalWeightedMeanPrecision(-0.5, 2.5))),
        ]

        @test_marginalrules [ with_float_conversions = true ] typeof(dot)(:in1_in2) [
            (input = (m_out = NormalMeanVariance(1.0, 2.0), m_in1 = PointMass(1.0), m_in2 = NormalMeanVariance(2.0, 2.0), meta = NoCorrection()), output = (in1 = PointMass(1.0), in2 = NormalWeightedMeanPrecision(1.5, 1.0))),
            (input = (m_out = NormalMeanVariance(1.0, 1.0), m_in1 = PointMass(2.0), m_in2 = NormalMeanPrecision(1.0, 0.5), meta = NoCorrection()), output = (in1 = PointMass(2.0), in2 = NormalWeightedMeanPrecision(2.5, 4.5))),
            (input = (m_out = NormalMeanVariance(1.0, 2.0), m_in1 = PointMass(-2.0), m_in2 = NormalWeightedMeanPrecision(0.5, 0.5), meta = NoCorrection()), output = (in1 = PointMass(-2.0), in2 = NormalWeightedMeanPrecision(-0.5, 2.5))),
        ]

        @test_marginalrules [ with_float_conversions = false ] typeof(dot)(:in1_in2) [
            (input = (m_out = NormalMeanVariance(1.0, 2.0), m_in1 = PointMass(1.0), m_in2 = NormalMeanVariance(2.0, 2.0), meta = TinyCorrection()), output = (in1 = PointMass(1.0), in2 = NormalWeightedMeanPrecision(1.5, 1.0))),
            (input = (m_out = NormalMeanVariance(1.0, 1.0), m_in1 = PointMass(2.0), m_in2 = NormalMeanPrecision(1.0, 0.5), meta = TinyCorrection()), output = (in1 = PointMass(2.0), in2 = NormalWeightedMeanPrecision(2.5, 4.5))),
            (input = (m_out = NormalMeanVariance(1.0, 2.0), m_in1 = PointMass(-2.0), m_in2 = NormalWeightedMeanPrecision(0.5, 0.5), meta = TinyCorrection()), output = (in1 = PointMass(-2.0), in2 = NormalWeightedMeanPrecision(-0.5, 2.5))),
        ]

    end

    @testset "in1_in2: (m_out::UnivariateNormalDistributionsFamily, m_in1::UnivariateNormalDistributionsFamily, m_in2::PointMass)" begin

        @test_marginalrules [ with_float_conversions = true ] typeof(dot)(:in1_in2) [
            (input = (m_out = NormalMeanVariance(1.0, 1.0), m_in1 = NormalMeanVariance(1.0, 2.0), m_in2 = PointMass(2.0), meta = NoCorrection()), output = (in1 = NormalWeightedMeanPrecision(2.5, 4.5), in2 = PointMass(2.0))),
            (input = (m_out = NormalMeanPrecision(3.0, 1.0), m_in1 = NormalMeanVariance(1.0, 2.0), m_in2 = PointMass(2.0), meta = NoCorrection()), output = (in1 = NormalWeightedMeanPrecision(6.5, 4.5), in2 = PointMass(2.0))),
            (input = (m_out = NormalWeightedMeanPrecision(4.0, 1.0), m_in1 = NormalMeanVariance(1.0, 3.0), m_in2 = PointMass(2.0), meta = NoCorrection()), output = (in1 = NormalWeightedMeanPrecision(25/3, 13/3), in2 = PointMass(2.0))),
        ]

        @test_marginalrules [ with_float_conversions = false ] typeof(dot)(:in1_in2) [
            (input = (m_out = NormalMeanVariance(1.0, 1.0), m_in1 = NormalMeanVariance(1.0, 2.0), m_in2 = PointMass(2.0), meta = TinyCorrection()), output = (in1 = NormalWeightedMeanPrecision(2.5, 4.5), in2 = PointMass(2.0))),
            (input = (m_out = NormalMeanPrecision(3.0, 1.0), m_in1 = NormalMeanVariance(1.0, 2.0), m_in2 = PointMass(2.0), meta = TinyCorrection()), output = (in1 = NormalWeightedMeanPrecision(6.5, 4.5), in2 = PointMass(2.0))),
            (input = (m_out = NormalWeightedMeanPrecision(4.0, 1.0), m_in1 = NormalMeanVariance(1.0, 3.0), m_in2 = PointMass(2.0), meta = TinyCorrection()), output = (in1 = NormalWeightedMeanPrecision(25/3, 13/3), in2 = PointMass(2.0))),
        ]

        ##

        @test_marginalrules [ with_float_conversions = true ] typeof(dot)(:in1_in2) [
            (input = (m_out = NormalMeanVariance(1.0, 1.0), m_in1 = NormalMeanVariance(1.0, 2.0), m_in2 = PointMass(2.0), meta = NoCorrection()), output = (in1 = NormalWeightedMeanPrecision(2.5, 4.5), in2 = PointMass(2.0))),
            (input = (m_out = NormalMeanVariance(3.0, 1.0), m_in1 = NormalMeanPrecision(1.0, 0.5), m_in2 = PointMass(2.0), meta = NoCorrection()), output = (in1 = NormalWeightedMeanPrecision(6.5, 4.5), in2 = PointMass(2.0))),
            (input = (m_out = NormalMeanVariance(4.0, 1.0), m_in1 = NormalWeightedMeanPrecision(1.0 / 3.0, 1.0 / 3.0), m_in2 = PointMass(2.0), meta = NoCorrection()), output = (in1 = NormalWeightedMeanPrecision(25/3, 13/3), in2 = PointMass(2.0))),
        ]

        @test_marginalrules [ with_float_conversions = false ] typeof(dot)(:in1_in2) [
            (input = (m_out = NormalMeanVariance(1.0, 1.0), m_in1 = NormalMeanVariance(1.0, 2.0), m_in2 = PointMass(2.0), meta = TinyCorrection()), output = (in1 = NormalWeightedMeanPrecision(2.5, 4.5), in2 = PointMass(2.0))),
            (input = (m_out = NormalMeanVariance(3.0, 1.0), m_in1 = NormalMeanPrecision(1.0, 0.5), m_in2 = PointMass(2.0), meta = TinyCorrection()), output = (in1 = NormalWeightedMeanPrecision(6.5, 4.5), in2 = PointMass(2.0))),
            (input = (m_out = NormalMeanVariance(4.0, 1.0), m_in1 = NormalWeightedMeanPrecision(1.0 / 3.0, 1.0 / 3.0), m_in2 = PointMass(2.0), meta = TinyCorrection()), output = (in1 = NormalWeightedMeanPrecision(25/3, 13/3), in2 = PointMass(2.0))),
        ]

    end

end

end