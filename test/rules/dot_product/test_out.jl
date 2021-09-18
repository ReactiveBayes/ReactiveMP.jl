module RulesDotProductOutTest

using Test
using ReactiveMP
using Random
import ReactiveMP: @test_rules
import LinearAlgebra: dot

@testset "rules:typeof(dot):out" begin

     @testset "Belief Propagation: (m_in1::PointMass{ <: AbstractVector }, m_in2::NormalDistributionsFamily)" begin

        @test_rules [ with_float_conversions = false ] typeof(dot)(:out, Marginalisation, symmetrical = [:in1, :in2]) [
            (input = (m_in1 = PointMass(1.0), m_in2 = NormalMeanVariance(2.0, 2.0)), output = NormalMeanVariance(4.0, 2.0)),
        ]

    end

#     @testset "Belief Propagation: (m_in1::NormalDistributionsFamily, m_in2::PointMass{ <: AbstractVector })" begin
#
#        @test_rules [ with_float_conversions = false ] typeof(dot)(:out, Marginalisation) [
#            (input = (m_in1 = NormalMeanVariance(2.0, 2.0), m_in2 = PointMass(1.0)), output = NormalMeanVariance(4.0, 2.0)),
#        ]
#
#    end

end
end