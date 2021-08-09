module AdditionNodeTest

using Test
using ReactiveMP
using Random
import ReactiveMP: @test_rules

@testset "AdditionNode" begin

    @testset "Creation" begin
        node = make_node(+)

        @test functionalform(node)          === +
        @test sdtype(node)                  === Deterministic()
        @test name.(interfaces(node))       === (:out, :in1, :in2)
        @test factorisation(node)           === ((1, 2, 3), )
        @test localmarginalnames(node)      === (:out_in1_in2, )
        @test metadata(node)                === nothing

        node = make_node(+, meta = 1)

        @test functionalform(node)          === +
        @test sdtype(node)                  === Deterministic()
        @test name.(interfaces(node))       === (:out, :in1, :in2)
        @test factorisation(node)           === ((1, 2, 3), )
        @test localmarginalnames(node)      === (:out_in1_in2, )
        @test metadata(node)                === 1
    end

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

            (input = (m_in1 = MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0]),  m_in2 = MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0])), output = MvNormalMeanCovariance([2.0, 6.0], [6.0 4.0; 4.0 8.0])),
            (input = (m_in1 = MvNormalMeanCovariance([-1.0, 3.0], [3.0 2.0; 2.0 4.0]),  m_in2 = MvNormalMeanCovariance([0.0, 3.0], [3.0 2.0; 2.0 4.0])), output = MvNormalMeanCovariance([-1.0, 6.0], [6.0 4.0; 4.0 8.0])),
            (input = (m_in1 = MvNormalMeanPrecision([1.0], [2.0]), m_in2 = MvNormalMeanPrecision([1.0], [2.0])), output = MvNormalMeanCovariance([2.0], [1.0])),
        ]
    end

     @testset "Update rules in1 PP" begin
        @test_rules [ with_float_conversions = false ] typeof(+)(:in1, Marginalisation) [
            (input = (m_out = PointMass(1.0),  m_in2 = PointMass(1.0)), output = PointMass(0.0)),
            (input = (m_out = PointMass([1.0]),  m_in2 = PointMass([-2.0])), output = PointMass([3.0])),
        ]
    end

    @testset "Update rules in1 PN" begin
        @test_rules [ with_float_conversions = false ] typeof(+)(:in1, Marginalisation) [
            (input = (m_out = NormalMeanVariance(1.0, 2.0),  m_in2 = PointMass(1.0)), output = NormalMeanVariance(0.0, 2.0)),
            (input = (m_out = NormalMeanVariance(1.0, 2.0),  m_in2 = PointMass(-3.0)), output = NormalMeanVariance(4.0, 2.0)),
            (input = (m_out = NormalMeanPrecision(1.0, 2.0),  m_in2 = PointMass(1.0)), output = NormalMeanPrecision(0.0, 2.0)),
            (input = (m_out = NormalMeanPrecision(1.0, 2.0),  m_in2 = PointMass(-3.0)), output = NormalMeanPrecision(4.0, 2.0)),
            (input = (m_out = MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0]),  m_in2 = PointMass([1.0, 1.0])), output = MvNormalMeanCovariance([0.0, 2.0], [3.0 2.0; 2.0 4.0])),
            (input = (m_out = MvNormalMeanCovariance([-4.0, 3.0], [3.0 2.0; 2.0 4.0]),  m_in2 = PointMass([1.0, 1.0])), output = MvNormalMeanCovariance([-5.0, 2.0], [3.0 2.0; 2.0 4.0])),
            (input = (m_out = MvNormalMeanPrecision([-4.0, 3.0], [3.0 2.0; 2.0 4.0]),  m_in2 = PointMass([1.0, 1.0])), output = MvNormalMeanPrecision([-5.0, 2.0], [3.0 2.0; 2.0 4.0])),
            (input = (m_out = MvNormalMeanPrecision([-4.0, 3.0], [3.0 2.0; 2.0 4.0]),  m_in2 = PointMass([-2.0, 1.0])), output = MvNormalMeanPrecision([-2.0, 2.0], [3.0 2.0; 2.0 4.0])),
        ]
    end

     @testset "Update rules in1 NN" begin
        @test_rules [ with_float_conversions = false ] typeof(+)(:in1, Marginalisation) [
            (input = (m_out = NormalMeanVariance(1.0, 2.0),  m_in2 = NormalMeanVariance(3.0, 4.0)), output = NormalMeanVariance(-2.0, 6.0)),
            (input = (m_out = NormalMeanVariance(-1.0, 2.0),  m_in2 = NormalMeanVariance(-2.0, 3.0)), output = NormalMeanVariance(1.0, 5.0)),
            (input = (m_out = NormalMeanPrecision(2.0, 2.0),  m_in2 = NormalMeanPrecision(-1.0, 3.0)), output = NormalMeanVariance(3.0, (2.0 + 3.0) / (2.0*3.0))),
            (input = (m_out = NormalMeanPrecision(-1.0, 2.0),  m_in2 = NormalMeanPrecision(-1.0, 3.0)), output = NormalMeanVariance(0.0, (2.0 + 3.0) / (2.0*3.0))),
            (input = (m_out = MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0]),  m_in2 = MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0])), output = MvNormalMeanCovariance([0.0, 0.0], [6.0 4.0; 4.0 8.0])),
            (input = (m_out = MvNormalMeanCovariance([-1.0, 3.0], [3.0 2.0; 2.0 4.0]),  m_in2 = MvNormalMeanCovariance([0.0, 3.0], [3.0 2.0; 2.0 4.0])), output = MvNormalMeanCovariance([-1.0, 0.0], [6.0 4.0; 4.0 8.0])),
            (input = (m_out = MvNormalMeanPrecision([1.0], [2.0]), m_in2 = MvNormalMeanPrecision([1.0], [2.0])), output = MvNormalMeanCovariance([0.0], [1.0])),
        ]
    end

    @testset "Update rules in2 PP" begin
        @test_rules [ with_float_conversions = false ] typeof(+)(:in2, Marginalisation) [
            (input = (m_out = PointMass(1.0),  m_in1 = PointMass(1.0)), output = PointMass(0.0)),
            (input = (m_out = PointMass([1.0]),  m_in1 = PointMass([-2.0])), output = PointMass([3.0])),
        ]
    end

    @testset "Update rules in2 PN" begin
        @test_rules [ with_float_conversions = false ] typeof(+)(:in2, Marginalisation) [
            (input = (m_out = NormalMeanVariance(1.0, 2.0),  m_in1 = PointMass(1.0)), output = NormalMeanVariance(0.0, 2.0)),
            (input = (m_out = NormalMeanVariance(1.0, 2.0),  m_in1 = PointMass(-3.0)), output = NormalMeanVariance(4.0, 2.0)),
            (input = (m_out = NormalMeanPrecision(1.0, 2.0),  m_in1 = PointMass(1.0)), output = NormalMeanPrecision(0.0, 2.0)),
            (input = (m_out = NormalMeanPrecision(1.0, 2.0),  m_in1 = PointMass(-3.0)), output = NormalMeanPrecision(4.0, 2.0)),
            (input = (m_out = MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0]),  m_in1 = PointMass([1.0, 1.0])), output = MvNormalMeanCovariance([0.0, 2.0], [3.0 2.0; 2.0 4.0])),
            (input = (m_out = MvNormalMeanCovariance([-4.0, 3.0], [3.0 2.0; 2.0 4.0]),  m_in1 = PointMass([1.0, 1.0])), output = MvNormalMeanCovariance([-5.0, 2.0], [3.0 2.0; 2.0 4.0])),
            (input = (m_out = MvNormalMeanPrecision([-4.0, 3.0], [3.0 2.0; 2.0 4.0]),  m_in1 = PointMass([1.0, 1.0])), output = MvNormalMeanPrecision([-5.0, 2.0], [3.0 2.0; 2.0 4.0])),
            (input = (m_out = MvNormalMeanPrecision([-4.0, 3.0], [3.0 2.0; 2.0 4.0]),  m_in1 = PointMass([-2.0, 1.0])), output = MvNormalMeanPrecision([-2.0, 2.0], [3.0 2.0; 2.0 4.0])),
        ]
    end

     @testset "Update rules in2 NN" begin
        @test_rules [ with_float_conversions = false ] typeof(+)(:in2, Marginalisation) [
            (input = (m_out = NormalMeanVariance(1.0, 2.0),  m_in1 = NormalMeanVariance(3.0, 4.0)), output = NormalMeanVariance(-2.0, 6.0)),
            (input = (m_out = NormalMeanVariance(-1.0, 2.0),  m_in1 = NormalMeanVariance(-2.0, 3.0)), output = NormalMeanVariance(1.0, 5.0)),
            (input = (m_out = NormalMeanPrecision(2.0, 2.0),  m_in1 = NormalMeanPrecision(-1.0, 3.0)), output = NormalMeanVariance(3.0, (2.0 + 3.0) / (2.0*3.0))),
            (input = (m_out = NormalMeanPrecision(-1.0, 2.0),  m_in1 = NormalMeanPrecision(-1.0, 3.0)), output = NormalMeanVariance(0.0, (2.0 + 3.0) / (2.0*3.0))),
            (input = (m_out = MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0]),  m_in1 = MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0])), output = MvNormalMeanCovariance([0.0, 0.0], [6.0 4.0; 4.0 8.0])),
            (input = (m_out = MvNormalMeanCovariance([-1.0, 3.0], [3.0 2.0; 2.0 4.0]),  m_in1 = MvNormalMeanCovariance([0.0, 3.0], [3.0 2.0; 2.0 4.0])), output = MvNormalMeanCovariance([-1.0, 0.0], [6.0 4.0; 4.0 8.0])),
            (input = (m_out = MvNormalMeanPrecision([1.0], [2.0]), m_in1 = MvNormalMeanPrecision([1.0], [2.0])), output = MvNormalMeanCovariance([0.0], [1.0])),
        ]
    end

end
end
