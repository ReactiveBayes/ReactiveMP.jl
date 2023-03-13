module RulesNormalMeanPrecisionOutTest

using Test
using ReactiveMP
using Random

import ReactiveMP: @test_rules

@testset "rules:NormalMeanPrecision:out" begin
    @testset "Belief Propagation: (m_μ::PointMass, m_τ::PointMass)" begin
        @test_rules [with_float_conversions = true] NormalMeanPrecision(:out, Marginalisation) [
            (input = (m_μ = PointMass(-1.0), m_τ = PointMass(2.0)), output = NormalMeanPrecision(-1.0, 2.0)),
            (input = (m_μ = PointMass(1.0), m_τ = PointMass(2.0)), output = NormalMeanPrecision(1.0, 2.0)),
            (input = (m_μ = PointMass(2.0), m_τ = PointMass(1.0)), output = NormalMeanPrecision(2.0, 1.0))
        ]
    end

    @testset "Belief Propagation: (m_μ::UnivariateNormalDistributionsFamily, m_τ::PointMass)" begin
        @test_rules [with_float_conversions = true] NormalMeanPrecision(:out, Marginalisation) [
            (input = (m_μ = NormalMeanPrecision(0.0, 1.0), m_τ = PointMass(2.0)), output = NormalMeanPrecision(0.0, 2.0 / 3.0)),
            (input = (m_μ = NormalMeanPrecision(-1.0, 1.0), m_τ = PointMass(1.5)), output = NormalMeanPrecision(-1.0, 0.6)),
            (input = (m_μ = NormalMeanPrecision(2.0, 0.5), m_τ = PointMass(1.0)), output = NormalMeanPrecision(2.0, 1.0 / 3.0))
        ]

        @test_rules [with_float_conversions = true] NormalMeanPrecision(:out, Marginalisation) [
            (input = (m_μ = NormalMeanVariance(0.0, 1.0), m_τ = PointMass(2.0)), output = NormalMeanPrecision(0.0, 2.0 / 3.0)),
            (input = (m_μ = NormalMeanVariance(-1.0, 1.0), m_τ = PointMass(1.5)), output = NormalMeanPrecision(-1.0, 0.6)),
            (input = (m_μ = NormalMeanVariance(2.0, 0.5), m_τ = PointMass(1.0)), output = NormalMeanPrecision(2.0, 2.0 / 3.0))
        ]

        @test_rules [with_float_conversions = true] NormalMeanPrecision(:out, Marginalisation) [
            (input = (m_μ = NormalWeightedMeanPrecision(0.0, 1.0), m_τ = PointMass(2.0)), output = NormalMeanPrecision(0.0, 2.0 / 3.0)),
            (input = (m_μ = NormalWeightedMeanPrecision(-1.0, 1.0), m_τ = PointMass(1.5)), output = NormalMeanPrecision(-1.0, 0.6)),
            (input = (m_μ = NormalWeightedMeanPrecision(2.0, 0.5), m_τ = PointMass(1.0)), output = NormalMeanPrecision(4.0, 1.0 / 3.0))
        ]
    end

    @testset "Variational: (m_μ::UnivariateNormalDistributionsFamily, q_τ::Any)" begin
        @test_rules [with_float_conversions = true] NormalMeanPrecision(:out, Marginalisation) [
            (input = (m_μ = NormalMeanPrecision(0.0, 1.0), q_τ = GammaShapeRate(2.0, 1.0)), output = NormalMeanPrecision(0.0, 2.0 / 3.0)),
            (input = (m_μ = NormalMeanPrecision(-1.0, 1.0), q_τ = GammaShapeRate(3.0, 2.0)), output = NormalMeanPrecision(-1.0, 0.6)),
            (input = (m_μ = NormalMeanPrecision(2.0, 0.5), q_τ = GammaShapeScale(10.0, 0.1)), output = NormalMeanPrecision(2.0, 1.0 / 3.0))
        ]

        @test_rules [with_float_conversions = true] NormalMeanPrecision(:out, Marginalisation) [
            (input = (m_μ = NormalMeanVariance(0.0, 1.0), q_τ = GammaShapeRate(2.0, 1.0)), output = NormalMeanPrecision(0.0, 2.0 / 3.0)),
            (input = (m_μ = NormalMeanVariance(-1.0, 1.0), q_τ = GammaShapeRate(3.0, 2.0)), output = NormalMeanPrecision(-1.0, 0.6)),
            (input = (m_μ = NormalMeanVariance(2.0, 0.5), q_τ = GammaShapeScale(10.0, 0.1)), output = NormalMeanPrecision(2.0, 2.0 / 3.0))
        ]

        @test_rules [with_float_conversions = true] NormalMeanPrecision(:out, Marginalisation) [
            (input = (m_μ = NormalWeightedMeanPrecision(0.0, 1.0), q_τ = GammaShapeRate(2.0, 1.0)), output = NormalMeanPrecision(0.0, 2.0 / 3.0)),
            (input = (m_μ = NormalWeightedMeanPrecision(-1.0, 1.0), q_τ = GammaShapeRate(3.0, 2.0)), output = NormalMeanPrecision(-1.0, 0.6)),
            (input = (m_μ = NormalWeightedMeanPrecision(2.0, 0.5), q_τ = GammaShapeScale(10.0, 0.1)), output = NormalMeanPrecision(4.0, 1.0 / 3.0))
        ]
    end

    @testset "Variational: (m_μ::PointMass, q_τ::Any)" begin
        @test_rules [with_float_conversions = true] NormalMeanPrecision(:out, Marginalisation) [
            (input = (m_μ = PointMass(-1.0), q_τ = GammaShapeRate(1.0, 1.0)), output = NormalMeanPrecision(-1.0, 1.0)),
            (input = (m_μ = PointMass(1.0), q_τ = GammaShapeScale(1.0, 1.0)), output = NormalMeanPrecision(1.0, 1.0)),
            (input = (m_μ = PointMass(2.0), q_τ = PointMass(1.0)), output = NormalMeanPrecision(2.0, 1.0))
        ]
    end

    @testset "Variational: (q_μ::Any, q_τ::Any)" begin
        @test_rules [with_float_conversions = true] NormalMeanPrecision(:out, Marginalisation) [
            (input = (q_μ = PointMass(-1.0), q_τ = PointMass(2.0)), output = NormalMeanPrecision(-1.0, 2.0)),
            (input = (q_μ = PointMass(1.0), q_τ = PointMass(2.0)), output = NormalMeanPrecision(1.0, 2.0)),
            (input = (q_μ = PointMass(2.0), q_τ = PointMass(1.0)), output = NormalMeanPrecision(2.0, 1.0))
        ]

        @test_rules [with_float_conversions = true] NormalMeanPrecision(:out, Marginalisation) [
            (input = (q_μ = NormalMeanVariance(-1.0, 2.0), q_τ = PointMass(2.0)), output = NormalMeanPrecision(-1.0, 2.0)),
            (input = (q_μ = NormalMeanPrecision(1.0, 4.0), q_τ = PointMass(3.0)), output = NormalMeanPrecision(1.0, 3.0)),
            (input = (q_μ = NormalWeightedMeanPrecision(2.0, 4.0), q_τ = PointMass(1.0)), output = NormalMeanPrecision(0.5, 1.0))
        ]

        @test_rules [with_float_conversions = true] NormalMeanPrecision(:out, Marginalisation) [
            (input = (q_μ = PointMass(-1.0), q_τ = Gamma(2.0, 1.0)), output = NormalMeanPrecision(-1.0, 2.0)),
            (input = (q_μ = PointMass(1.0), q_τ = Gamma(4.0, 2.0)), output = NormalMeanPrecision(1.0, 8.0)),
            (input = (q_μ = PointMass(2.0), q_τ = Gamma(4.0, 6.0)), output = NormalMeanPrecision(2.0, 24.0))
        ]
    end
end

end
