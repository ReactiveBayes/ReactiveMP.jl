module RulesNormalMeanPrecisionMeanTest

using Test
using ReactiveMP
using Random

import ReactiveMP: @test_rules

@testset "rules:NormalMeanPrecision:mean" begin
    @testset "Belief Propagation: (m_out::PointMass, m_τ::PointMass)" begin
        @test_rules [with_float_conversions = true] NormalMeanPrecision(:μ, Marginalisation) [
            (input = (m_out = PointMass(-1.0), m_τ = PointMass(2.0)), output = NormalMeanPrecision(-1.0, 2.0)),
            (input = (m_out = PointMass(1.0), m_τ = PointMass(2.0)), output = NormalMeanPrecision(1.0, 2.0)),
            (input = (m_out = PointMass(2.0), m_τ = PointMass(1.0)), output = NormalMeanPrecision(2.0, 1.0))
        ]
    end

    @testset "Belief Propagation: (m_out::UnivariateNormalDistributionsFamily, m_τ::PointMass)" begin
        @test_rules [with_float_conversions = true] NormalMeanPrecision(:μ, Marginalisation) [
            (input = (m_out = NormalMeanPrecision(0.0, 1.0), m_τ = PointMass(2.0)), output = NormalMeanVariance(0.0, 3.0 / 2.0)),
            (input = (m_out = NormalMeanPrecision(-1.0, 1.0), m_τ = PointMass(1.5)), output = NormalMeanVariance(-1.0, 10.0 / 6.0)),
            (input = (m_out = NormalMeanPrecision(2.0, 0.5), m_τ = PointMass(1.0)), output = NormalMeanVariance(2.0, 3.0 / 1.0))
        ]

        @test_rules [with_float_conversions = true] NormalMeanPrecision(:μ, Marginalisation) [
            (input = (m_out = NormalMeanVariance(0.0, 1.0), m_τ = PointMass(2.0)), output = NormalMeanVariance(0.0, 3.0 / 2.0)),
            (input = (m_out = NormalMeanVariance(-1.0, 1.0), m_τ = PointMass(1.5)), output = NormalMeanVariance(-1.0, 10.0 / 6.0)),
            (input = (m_out = NormalMeanVariance(2.0, 0.5), m_τ = PointMass(1.0)), output = NormalMeanVariance(2.0, 3.0 / 2.0))
        ]

        @test_rules [with_float_conversions = true] NormalMeanPrecision(:μ, Marginalisation) [
            (input = (m_out = NormalWeightedMeanPrecision(0.0, 1.0), m_τ = PointMass(2.0)), output = NormalMeanVariance(0.0, 3.0 / 2.0)),
            (input = (m_out = NormalWeightedMeanPrecision(-1.0, 1.0), m_τ = PointMass(1.5)), output = NormalMeanVariance(-1.0, 10.0 / 6.0)),
            (input = (m_out = NormalWeightedMeanPrecision(2.0, 0.5), m_τ = PointMass(1.0)), output = NormalMeanVariance(4.0, 3.0 / 1.0))
        ]
    end

    @testset "Variational: (m_out::UnivariateNormalDistributionsFamily, q_τ::Any)" begin
        @test_rules [with_float_conversions = true] NormalMeanPrecision(:μ, Marginalisation) [
            (input = (m_out = NormalMeanPrecision(0.0, 1.0), q_τ = GammaShapeRate(2.0, 1.0)), output = NormalMeanVariance(0.0, 3.0 / 2.0)),
            (input = (m_out = NormalMeanPrecision(-1.0, 1.0), q_τ = GammaShapeRate(3.0, 2.0)), output = NormalMeanVariance(-1.0, 10.0 / 6.0)),
            (input = (m_out = NormalMeanPrecision(2.0, 0.5), q_τ = GammaShapeScale(10.0, 0.1)), output = NormalMeanVariance(2.0, 3.0 / 1.0))
        ]

        @test_rules [with_float_conversions = true] NormalMeanPrecision(:μ, Marginalisation) [
            (input = (m_out = NormalMeanVariance(0.0, 1.0), q_τ = GammaShapeRate(2.0, 1.0)), output = NormalMeanVariance(0.0, 3.0 / 2.0)),
            (input = (m_out = NormalMeanVariance(-1.0, 1.0), q_τ = GammaShapeRate(3.0, 2.0)), output = NormalMeanVariance(-1.0, 10.0 / 6.0)),
            (input = (m_out = NormalMeanVariance(2.0, 0.5), q_τ = GammaShapeScale(10.0, 0.1)), output = NormalMeanVariance(2.0, 3.0 / 2.0))
        ]

        @test_rules [with_float_conversions = true] NormalMeanPrecision(:μ, Marginalisation) [
            (input = (m_out = NormalWeightedMeanPrecision(0.0, 1.0), q_τ = GammaShapeRate(2.0, 1.0)), output = NormalMeanVariance(0.0, 3.0 / 2.0)),
            (input = (m_out = NormalWeightedMeanPrecision(-1.0, 1.0), q_τ = GammaShapeRate(3.0, 2.0)), output = NormalMeanVariance(-1.0, 10.0 / 6.0)),
            (input = (m_out = NormalWeightedMeanPrecision(2.0, 0.5), q_τ = GammaShapeScale(10.0, 0.1)), output = NormalMeanVariance(4.0, 3.0 / 1.0))
        ]
    end

    @testset "Variational: (m_out::PointMass, q_τ::Any)" begin
        @test_rules [with_float_conversions = true] NormalMeanPrecision(:μ, Marginalisation) [
            (input = (m_out = PointMass(-1.0), q_τ = GammaShapeRate(1.0, 1.0)), output = NormalMeanPrecision(-1.0, 1.0)),
            (input = (m_out = PointMass(1.0), q_τ = GammaShapeScale(1.0, 1.0)), output = NormalMeanPrecision(1.0, 1.0)),
            (input = (m_out = PointMass(2.0), q_τ = PointMass(1.0)), output = NormalMeanPrecision(2.0, 1.0))
        ]
    end

    @testset "Variational: (q_out::Any, q_τ::Any)" begin
        @test_rules [with_float_conversions = true] NormalMeanPrecision(:μ, Marginalisation) [
            (input = (q_out = PointMass(-1.0), q_τ = PointMass(2.0)), output = NormalMeanPrecision(-1.0, 2.0)),
            (input = (q_out = PointMass(1.0), q_τ = PointMass(2.0)), output = NormalMeanPrecision(1.0, 2.0)),
            (input = (q_out = PointMass(2.0), q_τ = PointMass(1.0)), output = NormalMeanPrecision(2.0, 1.0))
        ]

        @test_rules [with_float_conversions = true] NormalMeanPrecision(:μ, Marginalisation) [
            (input = (q_out = NormalMeanVariance(-1.0, 2.0), q_τ = PointMass(2.0)), output = NormalMeanPrecision(-1.0, 2.0)),
            (input = (q_out = NormalMeanPrecision(1.0, 4.0), q_τ = PointMass(3.0)), output = NormalMeanPrecision(1.0, 3.0)),
            (input = (q_out = NormalWeightedMeanPrecision(2.0, 4.0), q_τ = PointMass(1.0)), output = NormalMeanPrecision(0.5, 1.0))
        ]

        @test_rules [with_float_conversions = true] NormalMeanPrecision(:μ, Marginalisation) [
            (input = (q_out = PointMass(-1.0), q_τ = Gamma(2.0, 1.0)), output = NormalMeanPrecision(-1.0, 2.0)),
            (input = (q_out = PointMass(1.0), q_τ = Gamma(4.0, 2.0)), output = NormalMeanPrecision(1.0, 8.0)),
            (input = (q_out = PointMass(2.0), q_τ = Gamma(4.0, 6.0)), output = NormalMeanPrecision(2.0, 24.0))
        ]
    end
end

end
