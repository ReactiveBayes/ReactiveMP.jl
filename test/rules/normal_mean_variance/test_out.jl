module RulesNormalMeanVarianceOutTest

using Test
using ReactiveMP
using Random
using Distributions

import ReactiveMP: @test_rules

@testset "rules:NormalMeanVariance:out" begin
    @testset "Belief Propagation: (m_μ::PointMass, m_v::PointMass)" begin
        @test_rules [with_float_conversions = true] NormalMeanVariance(:out, Marginalisation) [
            (input = (m_μ = PointMass(-1.0), m_v = PointMass(2.0)), output = NormalMeanVariance(-1.0, 2.0)),
            (input = (m_μ = PointMass(1.0), m_v = PointMass(2.0)), output = NormalMeanVariance(1.0, 2.0)),
            (input = (m_μ = PointMass(2.0), m_v = PointMass(1.0)), output = NormalMeanVariance(2.0, 1.0))
        ]
    end

    @testset "Belief Propagation: (m_μ::UnivariateNormalDistributionsFamily, m_v::PointMass)" begin
        @test_rules [with_float_conversions = true] NormalMeanVariance(:out, Marginalisation) [
            (input = (m_μ = NormalMeanVariance(0.0, 1.0), m_v = PointMass(2.0)), output = NormalMeanVariance(0.0, 3.0)),
            (input = (m_μ = NormalMeanVariance(-1.0, 1.0), m_v = PointMass(1.5)), output = NormalMeanVariance(-1.0, 2.5)),
            (input = (m_μ = NormalMeanVariance(2.0, 0.5), m_v = PointMass(1.0)), output = NormalMeanVariance(2.0, 1.5))
        ]

        @test_rules [with_float_conversions = true] NormalMeanVariance(:out, Marginalisation) [
            (input = (m_μ = NormalMeanPrecision(0.0, 1.0), m_v = PointMass(2.0)), output = NormalMeanVariance(0.0, 3.0)),
            (input = (m_μ = NormalMeanPrecision(-1.0, 1.0), m_v = PointMass(1.5)), output = NormalMeanVariance(-1.0, 2.5)),
            (input = (m_μ = NormalMeanPrecision(2.0, 0.5), m_v = PointMass(1.0)), output = NormalMeanVariance(2.0, 3.0))
        ]

        @test_rules [with_float_conversions = true] NormalMeanVariance(:out, Marginalisation) [
            (input = (m_μ = NormalWeightedMeanPrecision(0.0, 1.0), m_v = PointMass(2.0)), output = NormalMeanVariance(0.0, 3.0)),
            (input = (m_μ = NormalWeightedMeanPrecision(-1.0, 1.0), m_v = PointMass(1.5)), output = NormalMeanVariance(-1.0, 2.5)),
            (input = (m_μ = NormalWeightedMeanPrecision(2.0, 0.5), m_v = PointMass(1.0)), output = NormalMeanVariance(4.0, 3.0))
        ]
    end

    @testset "Belief Propagation: (m_μ::UnivariateNormalDistributionsFamily, q_v::PointMass)" begin
        @test_rules [with_float_conversions = true] NormalMeanVariance(:out, Marginalisation) [
            (input = (m_μ = NormalMeanVariance(0.0, 1.0), q_v = PointMass(2.0)), output = NormalMeanVariance(0.0, 3.0)),
            (input = (m_μ = NormalMeanVariance(-1.0, 1.0), q_v = PointMass(1.5)), output = NormalMeanVariance(-1.0, 2.5)),
            (input = (m_μ = NormalMeanVariance(2.0, 0.5), q_v = PointMass(1.0)), output = NormalMeanVariance(2.0, 1.5))
        ]

        @test_rules [with_float_conversions = true] NormalMeanVariance(:out, Marginalisation) [
            (input = (m_μ = NormalMeanPrecision(0.0, 1.0), q_v = PointMass(2.0)), output = NormalMeanVariance(0.0, 3.0)),
            (input = (m_μ = NormalMeanPrecision(-1.0, 1.0), q_v = PointMass(1.5)), output = NormalMeanVariance(-1.0, 2.5)),
            (input = (m_μ = NormalMeanPrecision(2.0, 0.5), q_v = PointMass(1.0)), output = NormalMeanVariance(2.0, 3.0))
        ]

        @test_rules [with_float_conversions = true] NormalMeanVariance(:out, Marginalisation) [
            (input = (m_μ = NormalWeightedMeanPrecision(0.0, 1.0), q_v = PointMass(2.0)), output = NormalMeanVariance(0.0, 3.0)),
            (input = (m_μ = NormalWeightedMeanPrecision(-1.0, 1.0), q_v = PointMass(1.5)), output = NormalMeanVariance(-1.0, 2.5)),
            (input = (m_μ = NormalWeightedMeanPrecision(2.0, 0.5), q_v = PointMass(1.0)), output = NormalMeanVariance(4.0, 3.0))
        ]
    end

    @testset "Variational: (m_μ::PointMass, q_v::Any)" begin
        @test_rules [with_float_conversions = true] NormalMeanVariance(:out, Marginalisation) [
            (input = (m_μ = PointMass(-1.0), q_v = GammaShapeRate(1.0, 1.0)), output = NormalMeanVariance(-1.0, 1.0)),
            (input = (m_μ = PointMass(1.0), q_v = GammaShapeScale(1.0, 1.0)), output = NormalMeanVariance(1.0, 1.0)),
            (input = (m_μ = PointMass(2.0), q_v = PointMass(1.0)), output = NormalMeanVariance(2.0, 1.0))
        ]
    end

    @testset "Variational: (q_μ::Any, q_v::Any)" begin
        @test_rules [with_float_conversions = true] NormalMeanVariance(:out, Marginalisation) [
            (input = (q_μ = PointMass(-1.0), q_v = PointMass(2.0)), output = NormalMeanVariance(-1.0, 2.0)),
            (input = (q_μ = PointMass(1.0), q_v = PointMass(2.0)), output = NormalMeanVariance(1.0, 2.0)),
            (input = (q_μ = PointMass(2.0), q_v = PointMass(1.0)), output = NormalMeanVariance(2.0, 1.0))
        ]

        @test_rules [with_float_conversions = true] NormalMeanVariance(:out, Marginalisation) [
            (input = (q_μ = NormalMeanVariance(-1.0, 2.0), q_v = PointMass(2.0)), output = NormalMeanVariance(-1.0, 2.0)),
            (input = (q_μ = NormalMeanPrecision(1.0, 4.0), q_v = PointMass(3.0)), output = NormalMeanVariance(1.0, 3.0)),
            (input = (q_μ = NormalWeightedMeanPrecision(2.0, 4.0), q_v = PointMass(1.0)), output = NormalMeanVariance(0.5, 1.0))
        ]

        @test_rules [with_float_conversions = true] NormalMeanVariance(:out, Marginalisation) [
            (input = (q_μ = PointMass(-1.0), q_v = InverseGamma(2.0, 1.0)), output = NormalMeanVariance(-1.0, 1.0)),
            (input = (q_μ = PointMass(1.0), q_v = InverseGamma(4.0, 2.0)), output = NormalMeanVariance(1.0, 2.0 / 3.0)),
            (input = (q_μ = PointMass(2.0), q_v = InverseGamma(4.0, 6.0)), output = NormalMeanVariance(2.0, 2.0))
        ]
    end
end

end
