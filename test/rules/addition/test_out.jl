module AdditionNodeTest

using Test
using ReactiveMP
using Random
import ReactiveMP: @test_rules

@testset "Addition out" begin
     @testset "Update rules out PP" begin
        @test_rules [ with_float_conversions = false ] typeof(+)(:out, Marginalisation) [
            (input = (m_in1 = PointMass(1.0),  m_in2 = PointMass(1.0)), output = PointMass(2.0)),
            (input = (m_in1 = PointMass([1.0]),  m_in2 = PointMass([-2.0])), output = PointMass([-1.0])),
        ]
    end

    @testset "Update rules out PN" begin
        @test_rules [ with_float_conversions = false ] typeof(+)(:out, Marginalisation) [
            (input = (m_in1 = NormalMeanVariance(1.0, 2.0),  m_in2 = PointMass(1.0)), output = NormalMeanVariance(2.0, 2.0)),
            (input = (m_in1 = NormalMeanVariance(1.0, 2.0),  m_in2 = PointMass(-3.0)), output = NormalMeanVariance(-2.0, 2.0)),
            (input = (m_in1 = NormalMeanPrecision(1.0, 2.0),  m_in2 = PointMass(1.0)), output = NormalMeanPrecision(2.0, 2.0)),
            (input = (m_in1 = NormalMeanPrecision(1.0, 2.0),  m_in2 = PointMass(-3.0)), output = NormalMeanPrecision(-2.0, 2.0)),
            (input = (m_in1 = MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0]),  m_in2 = PointMass([1.0, 1.0])), output = MvNormalMeanCovariance([2.0, 4.0], [3.0 2.0; 2.0 4.0])),
            (input = (m_in1 = MvNormalMeanCovariance([-4.0, 3.0], [3.0 2.0; 2.0 4.0]),  m_in2 = PointMass([1.0, 1.0])), output = MvNormalMeanCovariance([-3.0, 4.0], [3.0 2.0; 2.0 4.0])),

            ##
            (input = (m_in1 = MvNormalMeanPrecision([-4.0, 3.0], [3.0 2.0; 2.0 4.0]),  m_in2 = PointMass([1.0, 1.0])), output = MvNormalMeanPrecision([-3.0, 4.0], [3.0 2.0; 2.0 4.0])),
            (input = (m_in1 = MvNormalMeanPrecision([-4.0, 3.0], [3.0 2.0; 2.0 4.0]),  m_in2 = PointMass([-2.0, 1.0])), output = MvNormalMeanPrecision([-6.0, 2.0], [3.0 2.0; 2.0 4.0])),
            ##
        ]
    end

     @testset "Update rules out NN" begin
        @test_rules [ with_float_conversions = false ] typeof(+)(:out, Marginalisation) [
            (input = (m_in1 = NormalMeanVariance(1.0, 2.0),  m_in2 = NormalMeanVariance(3.0, 4.0)), output = NormalMeanVariance(4.0, 6.0)),
            (input = (m_in1 = NormalMeanVariance(-1.0, 2.0),  m_in2 = NormalMeanVariance(-2.0, 3.0)), output = NormalMeanVariance(-3.0, 5.0)),

            ##
            (input = (m_in1 = NormalMeanPrecision(2.0, 2.0),  m_in2 = NormalMeanPrecision(-1.0, 3.0)), output = NormalMeanVariance(1.0, (2.0 + 3.0) / (2.0*3.0))),
            (input = (m_in1 = NormalMeanPrecision(-1.0, 2.0),  m_in2 = NormalMeanPrecision(-1.0, 3.0)), output = NormalMeanVariance(-2.0, (2.0 + 3.0) / (2.0*3.0))),
            ##

            (input = (m_in1 = NormalMeanPrecision(2.0, 2.0),  m_in2 = NormalMeanVariance(-1.0, 3.0)), output = NormalMeanVariance(1.0, 3.5)),
            (input = (m_in1 = MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0]),  m_in2 = MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0])), output = MvNormalMeanCovariance([2.0, 6.0], [6.0 4.0; 4.0 8.0])),
            (input = (m_in1 = MvNormalMeanCovariance([-1.0, 3.0], [3.0 2.0; 2.0 4.0]),  m_in2 = MvNormalMeanCovariance([0.0, 3.0], [3.0 2.0; 2.0 4.0])), output = MvNormalMeanCovariance([-1.0, 6.0], [6.0 4.0; 4.0 8.0])),
            (input = (m_in1 = MvNormalMeanPrecision([1.0], [2.0]), m_in2 = MvNormalMeanPrecision([1.0], [2.0])), output = MvNormalMeanCovariance([2.0], [1.0])),
            (input = (m_in1 = MvNormalMeanCovariance([1.0], [2.0]), m_in2 = MvNormalMeanPrecision([1.0], [2.0])), output = MvNormalMeanCovariance([2.0], [2.5])),
        ]
    end
end
end