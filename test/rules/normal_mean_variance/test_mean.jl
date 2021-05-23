module RulesNormalMeanVarianceMeanTest

using Test
using ReactiveMP
using Random

import ReactiveMP: @test_rules

@testset "rules:NormalMeanVariance:mean" begin

    @testset "Belief Propagation: (m_out::PointMass, m_v::PointMass)" begin

        @test_rules [ with_float_conversions = true ] NormalMeanVariance(:μ, Marginalisation) [
            (input = (m_out = PointMass(-1.0), m_v = PointMass(2.0)), output = NormalMeanVariance(-1.0, 2.0)),
            (input = (m_out = PointMass(1.0), m_v = PointMass(2.0)),  output = NormalMeanVariance(1.0, 2.0)),
            (input = (m_out = PointMass(2.0), m_v = PointMass(1.0)),  output = NormalMeanVariance(2.0, 1.0))
        ]

    end

    @testset "Belief Propagation: (m_out::UnivariateNormalDistributionsFamily, m_v::PointMass)" begin

        @test_rules [ with_float_conversions = true ] NormalMeanVariance(:μ, Marginalisation) [
            (input = (m_out = NormalMeanVariance(0.0, 1.0),  m_v = PointMass(2.0)), output = NormalMeanVariance(0.0, 3.0)),
            (input = (m_out = NormalMeanVariance(-1.0, 1.0), m_v = PointMass(1.5)), output = NormalMeanVariance(-1.0, 2.5)),
            (input = (m_out = NormalMeanVariance(2.0, 0.5),  m_v = PointMass(1.0)), output = NormalMeanVariance(2.0, 1.5)),
        ]

        @test_rules [ with_float_conversions = true ] NormalMeanVariance(:μ, Marginalisation) [
            (input = (m_out = NormalMeanPrecision(0.0, 1.0),  m_v = PointMass(2.0)), output = NormalMeanVariance(0.0, 3.0)),
            (input = (m_out = NormalMeanPrecision(-1.0, 1.0), m_v = PointMass(1.5)), output = NormalMeanVariance(-1.0, 2.5)),
            (input = (m_out = NormalMeanPrecision(2.0, 0.5),  m_v = PointMass(1.0)), output = NormalMeanVariance(2.0, 3.0)),
        ]

        @test_rules [ with_float_conversions = true ] NormalMeanVariance(:μ, Marginalisation) [
            (input = (m_out = NormalWeightedMeanPrecision(0.0, 1.0),  m_v = PointMass(2.0)), output = NormalMeanVariance(0.0, 3.0)),
            (input = (m_out = NormalWeightedMeanPrecision(-1.0, 1.0), m_v = PointMass(1.5)), output = NormalMeanVariance(-1.0, 2.5)),
            (input = (m_out = NormalWeightedMeanPrecision(2.0, 0.5),  m_v = PointMass(1.0)), output = NormalMeanVariance(4.0, 3.0)),
        ]

    end

end



end