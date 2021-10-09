module RulesDotProductOutTest

using Test
using ReactiveMP
using Random
import ReactiveMP: @test_rules
import LinearAlgebra: dot

@testset "rules:typeof(dot):out" begin

     @testset "Belief Propagation: (m_in1::PointMass, m_in2::NormalDistributionsFamily)" begin

        @test_rules [ with_float_conversions = true ] typeof(dot)(:out, Marginalisation) [
            (input = (m_in1 = PointMass(1.0), m_in2 = NormalMeanVariance(2.0, 2.0)), output = NormalMeanVariance(2.0, 2.0)),
            (input = (m_in1 = PointMass(-1.0), m_in2 = NormalMeanVariance(3.0, 1.0)), output = NormalMeanVariance(-3.0, 1.0)),
            (input = (m_in1 = PointMass(2.0), m_in2 = NormalMeanVariance(4.0, 2.0)), output = NormalMeanVariance(8.0, 8.0)),
        ]

    end

    @testset "Belief Propagation: (m_in1::NormalDistributionsFamily, m_in2::PointMass)" begin

        @test_rules [ with_float_conversions = true ] typeof(dot)(:out, Marginalisation) [
            (input = (m_in1 = NormalMeanVariance(2.0, 2.0), m_in2 = PointMass(4.0)), output = NormalMeanVariance(8.0, 32.0)),
            (input = (m_in1 = NormalMeanVariance(2.0, 3.0), m_in2 = PointMass(2.0)), output = NormalMeanVariance(4.0, 12.0)),
            (input = (m_in1 = NormalMeanVariance(4.0, 2.0), m_in2 = PointMass(1.0)), output = NormalMeanVariance(4.0, 2.0)),
        ]

    end

    @testset "Belief Propagation: (m_in1::NormalDistributionsFamily, m_in2::PointMass{ <: AbstractVector })" begin

        @test_rules [ with_float_conversions = true ] typeof(dot)(:out, Marginalisation) [
            (input = (m_in1 = MvNormalMeanCovariance([-1.0, 1.0], [1.0 -1.0; -3.0 2.0]), m_in2 = PointMass([1.0, 1.0])), output = NormalMeanVariance(0.0, -1.0)),
            (input = (m_in1 = MvNormalMeanCovariance([2.0, 1.0], [2.0 -1.0; -3.0 1.0]), m_in2 = PointMass([1.0, 1.0])), output = NormalMeanVariance(3.0, -1.0)),
            (input = (m_in1 = MvNormalMeanCovariance([3.0, 2.0], [1.0 -2.0; -2.0 1.0]), m_in2 = PointMass([1.0, 1.0])), output = NormalMeanVariance(5.0, -2.0)),
        ]

    end

    @testset "Belief Propagation: (m_in1::PointMass{ <: AbstractVector }, m_in2::NormalDistributionsFamily)" begin

        @test_rules [ with_float_conversions = true ] typeof(dot)(:out, Marginalisation) [
            (input = (m_in1 = PointMass([4.0, 1.0]), m_in2 =  MvNormalMeanCovariance([2.0, 1.0], [1.0 -2.0; -2.0 1.0])), output = NormalMeanVariance(9.0, 1.0)),
            (input = (m_in1 = PointMass([2.0, 2.0]), m_in2 =  MvNormalMeanCovariance([-1.0, 1.0], [3.0 -3.0; -3.0 3.0])), output = NormalMeanVariance(0.0, 0.0)),
            (input = (m_in1 = PointMass([1.0, 3.0]), m_in2 =  MvNormalMeanCovariance([-3.0, 1.0], [2.0 -2.0; -2.0 2.0])), output = NormalMeanVariance(0.0, 8.0)),
        ]

    end

end
end