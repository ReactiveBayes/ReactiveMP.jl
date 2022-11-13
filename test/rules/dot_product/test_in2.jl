module RulesDotProductIn2Test

using Test
using ReactiveMP
using Random
import ReactiveMP: @test_rules
import LinearAlgebra: dot

@testset "rules:typeof(dot):in2" begin
    @testset "Belief Propagation: (m_out::UnivariateNormalDistributionsFamily, m_in1::PointMass)" begin
        @test_rules [with_float_conversions = true] typeof(dot)(:in2, Marginalisation) [
            (input = (m_out = NormalMeanVariance(2.0, 2.0), m_in1 = PointMass(-1.0), meta = NoCorrection()), output = NormalWeightedMeanPrecision(-1.0, 0.5)),
            (input = (m_out = NormalMeanPrecision(1.0, 1.0), m_in1 = PointMass(-2.0), meta = NoCorrection()), output = NormalWeightedMeanPrecision(-2.0, 4.0)),
            (input = (m_out = NormalWeightedMeanPrecision(2.0, 1.0), m_in1 = PointMass(-1.0), meta = NoCorrection()), output = NormalWeightedMeanPrecision(-2.0, 1.0))
        ]

        @test_rules [with_float_conversions = true] typeof(dot)(:in2, Marginalisation) [
            (input = (m_out = NormalMeanVariance(2.0, 2.0), m_in1 = PointMass(-1.0), meta = TinyCorrection()), output = NormalWeightedMeanPrecision(-1.0, 0.5)),
            (input = (m_out = NormalMeanPrecision(1.0, 1.0), m_in1 = PointMass(-2.0), meta = TinyCorrection()), output = NormalWeightedMeanPrecision(-2.0, 4.0)),
            (input = (m_out = NormalWeightedMeanPrecision(2.0, 1.0), m_in1 = PointMass(-1.0), meta = TinyCorrection()), output = NormalWeightedMeanPrecision(-2.0, 1.0))
        ]

        @test_rules [with_float_conversions = true] typeof(dot)(:in2, Marginalisation) [
            (input = (m_out = NormalMeanVariance(2.0, 2.0), m_in1 = PointMass(0.0), meta = NoCorrection()), output = NormalWeightedMeanPrecision(0.0, 0.0)),
            (input = (m_out = NormalMeanPrecision(1.0, 1.0), m_in1 = PointMass(0.0), meta = NoCorrection()), output = NormalWeightedMeanPrecision(0.0, 0.0)),
            (input = (m_out = NormalWeightedMeanPrecision(2.0, 1.0), m_in1 = PointMass(0.0), meta = NoCorrection()), output = NormalWeightedMeanPrecision(0.0, 0.0))
        ]

        @test_rules [with_float_conversions = true] typeof(dot)(:in2, Marginalisation) [
            (input = (m_out = NormalMeanVariance(2.0, 2.0), m_in1 = PointMass(0.0), meta = TinyCorrection()), output = NormalWeightedMeanPrecision(0.0, tiny)),
            (input = (m_out = NormalMeanPrecision(1.0, 1.0), m_in1 = PointMass(0.0), meta = TinyCorrection()), output = NormalWeightedMeanPrecision(0.0, tiny)),
            (input = (m_out = NormalWeightedMeanPrecision(2.0, 1.0), m_in1 = PointMass(0.0), meta = TinyCorrection()), output = NormalWeightedMeanPrecision(0.0, tiny))
        ]
    end

    @testset "Belief Propagation: (m_out::UnivariateNormalDistributionsFamily, m_in1::PointMass)" begin
        @test_rules [with_float_conversions = true] typeof(dot)(:in2, Marginalisation) [
            (
                input = (m_out = NormalMeanVariance(2.0, 2.0), m_in1 = PointMass([-1.0, 1.0]), meta = NoCorrection()),
                output = MvNormalWeightedMeanPrecision([-1.0, 1.0], [0.5 -0.5; -0.5 0.5])
            ),
            (
                input = (m_out = NormalMeanPrecision(1.0, inv(2.0)), m_in1 = PointMass([2.0, 1.0]), meta = NoCorrection()),
                output = MvNormalWeightedMeanPrecision([1.0, 0.5], [2.0 1.0; 1.0 0.5])
            ),
            (
                input = (m_out = NormalWeightedMeanPrecision(1.0, 1.0), m_in1 = PointMass([-1.0, 3.0]), meta = NoCorrection()),
                output = MvNormalWeightedMeanPrecision([-1.0, 3.0], [1.0 -3.0; -3.0 9.0])
            )
        ]

        @test_rules [with_float_conversions = false] typeof(dot)(:in2, Marginalisation) [
            (
                input = (m_out = NormalMeanVariance(2.0, 2.0), m_in1 = PointMass([-1.0, 1.0]), meta = TinyCorrection()),
                output = MvNormalWeightedMeanPrecision([-1.0, 1.0], [0.5+tiny -0.5; -0.5 0.5+tiny])
            ),
            (
                input = (m_out = NormalMeanPrecision(1.0, inv(2.0)), m_in1 = PointMass([2.0, 1.0]), meta = TinyCorrection()),
                output = MvNormalWeightedMeanPrecision([1.0, 0.5], [2.0+tiny 1.0; 1.0 0.5+tiny])
            ),
            (
                input = (m_out = NormalWeightedMeanPrecision(1.0, 1.0), m_in1 = PointMass([-1.0, 3.0]), meta = TinyCorrection()),
                output = MvNormalWeightedMeanPrecision([-1.0, 3.0], [1.0+tiny -3.0; -3.0 9.0+tiny])
            )
        ]
    end
end

end
