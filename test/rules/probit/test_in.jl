module RulesProbitInTest

using Test
using ReactiveMP
using Random
using Distributions
using StatsFuns: normcdf

import ReactiveMP: @test_rules


@testset "rules:Probit:in" begin

    @testset "Belief Propagation: (m_out::PointMass, )" begin
        
        m_out = PointMass(1)
        m_in = ReactiveMP.@call_rule(Probit(:in, Marginalisation), (m_out=m_out,))
        @test typeof(m_in) <: ContinuousUnivariateLogPdf

    end

    @testset "Expectation Propagation: (m_out::PointMass, m_in::UnivariateNormalDistributionsFamily)" begin
        
        @test_rules [ with_float_conversions = true ] Probit(:in, Marginalisation) [
            (input = (m_out = PointMass(1.0), m_in = NormalMeanVariance(1.0, 0.5) ), output = NormalWeightedMeanPrecision(0.672361658269399, 0.329500399396070) ), 
            (input = (m_out = PointMass(1.0), m_in = NormalMeanPrecision(1.0, 2.0) ), output = NormalWeightedMeanPrecision(0.672361658269399, 0.329500399396070) ), 
            (input = (m_out = PointMass(1.0), m_in = NormalWeightedMeanPrecision(2.0, 2.0) ), output = NormalWeightedMeanPrecision(0.672361658269399, 0.329500399396070) ),
            (input = (m_out = PointMass(0.0), m_in = NormalMeanVariance(1.0, 0.5) ), output = NormalWeightedMeanPrecision(-0.821224653874111, 0.700344737736008) ), 
            (input = (m_out = PointMass(0.0), m_in = NormalMeanPrecision(1.0, 2.0) ), output = NormalWeightedMeanPrecision(-0.821224653874111, 0.700344737736008) ), 
            (input = (m_out = PointMass(0.0), m_in = NormalWeightedMeanPrecision(2.0, 2.0) ), output = NormalWeightedMeanPrecision(-0.821224653874111, 0.700344737736008) )
        ]

    end

    @testset "Expectation Propagation: (m_out::Bernoulli, m_in::UnivariateNormalDistributionsFamily)" begin
        
        @test_rules [ with_float_conversions = true ] Probit(:in, Marginalisation) [
            (input = (m_out = Bernoulli(1.0), m_in = NormalMeanVariance(1.0, 0.5) ), output = NormalWeightedMeanPrecision(0.672361658269399, 0.329500399396070) ), 
            (input = (m_out = Bernoulli(1.0), m_in = NormalMeanVariance(1.0, 0.5) ), output = NormalWeightedMeanPrecision(0.672361658269399, 0.329500399396070) ), 
            (input = (m_out = Bernoulli(1.0), m_in = NormalMeanVariance(1.0, 0.5) ), output = NormalWeightedMeanPrecision(0.672361658269399, 0.329500399396070) ), 
            (input = (m_out = Bernoulli(0.8), m_in = NormalMeanVariance(1.0, 0.5) ), output = NormalWeightedMeanPrecision(0.427017495944859, 0.199141999223396) ), 
            (input = (m_out = Bernoulli(0.8), m_in = NormalMeanPrecision(1.0, 2.0) ), output = NormalWeightedMeanPrecision(0.427017495944859, 0.199141999223396) ), 
            (input = (m_out = Bernoulli(0.8), m_in = NormalWeightedMeanPrecision(2.0, 2.0) ), output = NormalWeightedMeanPrecision(0.427017495944859, 0.199141999223396) ), 
            (input = (m_out = Bernoulli(0.5), m_in = NormalMeanVariance(1.0, 0.5) ), output = NormalWeightedMeanPrecision(4.0e-12, 4.0e-12) ), 
            (input = (m_out = Bernoulli(0.5), m_in = NormalMeanPrecision(1.0, 2.0) ), output = NormalWeightedMeanPrecision(4.0e-12, 4.0e-12) ), 
            (input = (m_out = Bernoulli(0.5), m_in = NormalWeightedMeanPrecision(2.0, 2.0) ), output = NormalWeightedMeanPrecision(4.0e-12, 4.0e-12) ),
            (input = (m_out = Bernoulli(0.0), m_in = NormalMeanVariance(1.0, 0.5) ), output = NormalWeightedMeanPrecision(-0.821224653874111, 0.700344737736008) ), 
            (input = (m_out = Bernoulli(0.0), m_in = NormalMeanPrecision(1.0, 2.0) ), output = NormalWeightedMeanPrecision(-0.821224653874111, 0.700344737736008) ), 
            (input = (m_out = Bernoulli(0.0), m_in = NormalWeightedMeanPrecision(2.0, 2.0) ), output = NormalWeightedMeanPrecision(-0.821224653874111, 0.700344737736008) )
        ]
        
    end 

end


end