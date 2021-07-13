module RulesProbitInTest

using Test
using ReactiveMP
using Random
using Distributions
using StatsFuns: normcdf

import ReactiveMP: @test_rules


@testset "rules:Probit:in" begin

    @testset "Belief Propagation: (m_out::PointMass, )" begin

        @test_rules [ with_float_conversions = true ] Probit(:in, Marginalisation) [
            (input = (m_out = PointMass(1.0), ), output = ContinuousUnivariateLogPdf((z) -> log(normcdf(z)))),
            (input = (m_out = PointMass(0.8), ), output = ContinuousUnivariateLogPdf((z) -> log(0.2 + 0.6*normcdf(z)))),
            (input = (m_out = PointMass(0.5), ), output = ContinuousUnivariateLogPdf((z) -> log(0.5))),
            (input = (m_out = PointMass(0.0), ), output = ContinuousUnivariateLogPdf((z) -> log(1 - normcdf(z)))),
        ]

    end

    @testset "Belief Propagation: (m_out::Missing, )" begin

        @test_rules [ with_float_conversions = true ] Probit(:in, Marginalisation) [
            (input = (m_out = Missing(), ), output = NormalWeightedMeanPrecision(0.0, tiny) )
        ]

    end

    @testset "Expectation Propagation: (m_out::PointMass, m_in::UnivariateNormalDistributionsFamily)" begin
        
        @test_rules [ with_float_conversions = true ] Probit(:in, MomentMatching) [
            (input = (m_out = PointMass(1.0), m_in = NormalMeanVariance(1.0, 0.5) ), output = NormalWeightedMeanPrecision(0.672361658269399, 0.329500399396070) ), 
            (input = (m_out = PointMass(1.0), m_in = NormalMeanPrecision(1.0, 2.0) ), output = NormalWeightedMeanPrecision(0.672361658269399, 0.329500399396070) ), 
            (input = (m_out = PointMass(1.0), m_in = NormalWeightedMeanPrecision(2.0, 2.0) ), output = NormalWeightedMeanPrecision(0.672361658269399, 0.329500399396070) ),
            (input = (m_out = PointMass(0.0), m_in = NormalMeanVariance(1.0, 0.5) ), output = NormalWeightedMeanPrecision(-0.821224653874111, 0.700344737736008) ), 
            (input = (m_out = PointMass(0.0), m_in = NormalMeanPrecision(1.0, 2.0) ), output = NormalWeightedMeanPrecision(-0.821224653874111, 0.700344737736008) ), 
            (input = (m_out = PointMass(0.0), m_in = NormalWeightedMeanPrecision(2.0, 2.0) ), output = NormalWeightedMeanPrecision(-0.821224653874111, 0.700344737736008) )
        ]

    end

    @testset "Expectation Propagation: (m_out::Bernoulli, m_in::UnivariateNormalDistributionsFamily)" begin
        
        @test_rules [ with_float_conversions = true ] Probit(:in, MomentMatching) [
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