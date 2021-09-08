module RulesProbitInTest

using Test
using ReactiveMP
using Random
using Distributions
using StatsFuns: normcdf, normccdf, normlogcdf, normlogccdf, normlogpdf, normpdf, logsumexp

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

    @testset "Expectation Propagation: (m_out::PointMass, m_in::UnivariateNormalDistributionsFamily)" begin
        
        @test_rules [ with_float_conversions = true ] Probit(:in, MomentMatching) [
            (input = (m_out = PointMass(1.0), m_in = NormalMeanVariance(1.0, 0.5) ), output = NormalWeightedMeanPrecision(0.6723616582693972, 0.32950039939606945) ), 
            (input = (m_out = PointMass(1.0), m_in = NormalMeanPrecision(1.0, 2.0) ), output = NormalWeightedMeanPrecision(0.6723616582693972, 0.32950039939606945) ), 
            (input = (m_out = PointMass(1.0), m_in = NormalWeightedMeanPrecision(2.0, 2.0) ), output = NormalWeightedMeanPrecision(0.6723616582693972, 0.32950039939606945) ),
            (input = (m_out = PointMass(0.0), m_in = NormalMeanVariance(1.0, 0.5) ), output = NormalWeightedMeanPrecision(-0.821224653874111, 0.7003447377360019) ), 
            (input = (m_out = PointMass(0.0), m_in = NormalMeanPrecision(1.0, 2.0) ), output = NormalWeightedMeanPrecision(-0.821224653874111, 0.7003447377360019) ), 
            (input = (m_out = PointMass(0.0), m_in = NormalWeightedMeanPrecision(2.0, 2.0) ), output = NormalWeightedMeanPrecision(-0.821224653874111, 0.7003447377360019) )
        ]

    end

    @testset "Expectation Propagation: (m_out::Bernoulli, m_in::UnivariateNormalDistributionsFamily)" begin
        
        @test_rules [ with_float_conversions = true ] Probit(:in, MomentMatching) [
            (input = (m_out = Bernoulli(1.0), m_in = NormalMeanVariance(1.0, 0.5) ), output = NormalWeightedMeanPrecision(0.6723616582693972, 0.32950039939606945) ), 
            (input = (m_out = Bernoulli(1.0), m_in = NormalMeanVariance(1.0, 0.5) ), output = NormalWeightedMeanPrecision(0.6723616582693972, 0.32950039939606945) ), 
            (input = (m_out = Bernoulli(1.0), m_in = NormalMeanVariance(1.0, 0.5) ), output = NormalWeightedMeanPrecision(0.6723616582693972, 0.32950039939606945) ), 
            (input = (m_out = Bernoulli(0.8), m_in = NormalMeanVariance(1.0, 0.5) ), output = NormalWeightedMeanPrecision(0.427017495944859, 0.199141999223396) ), 
            (input = (m_out = Bernoulli(0.8), m_in = NormalMeanPrecision(1.0, 2.0) ), output = NormalWeightedMeanPrecision(0.427017495944859, 0.199141999223396) ), 
            (input = (m_out = Bernoulli(0.8), m_in = NormalWeightedMeanPrecision(2.0, 2.0) ), output = NormalWeightedMeanPrecision(0.427017495944859, 0.199141999223396) ), 
            (input = (m_out = Bernoulli(0.0), m_in = NormalMeanVariance(1.0, 0.5) ), output = NormalWeightedMeanPrecision(-0.821224653874111, 0.7003447377360019) ), 
            (input = (m_out = Bernoulli(0.0), m_in = NormalMeanPrecision(1.0, 2.0) ), output = NormalWeightedMeanPrecision(-0.821224653874111, 0.7003447377360019) ), 
            (input = (m_out = Bernoulli(0.0), m_in = NormalWeightedMeanPrecision(2.0, 2.0) ), output = NormalWeightedMeanPrecision(-0.821224653874111, 0.7003447377360019) )
        ]

        # Test against an extreme case with m_out = Bernoulli(0.5)

        @test_rules [ with_float_conversions = false, atol = 1e-13 ] Probit(:in, MomentMatching) [
            (input = (m_out = Bernoulli(0.5), m_in = NormalMeanVariance(1.0, 0.5) ), output = NormalWeightedMeanPrecision(4.0 * tiny, 4.0 * tiny) ), 
            (input = (m_out = Bernoulli(0.5), m_in = NormalMeanPrecision(1.0, 2.0) ), output = NormalWeightedMeanPrecision(4.0 * tiny, 4.0 * tiny) ), 
            (input = (m_out = Bernoulli(0.5), m_in = NormalWeightedMeanPrecision(2.0, 2.0) ), output = NormalWeightedMeanPrecision(4.0 * tiny, 4.0 * tiny) ),
        ]

        @test_rules [ with_float_conversions = false, atol = 1e-7 ] Probit(:in, MomentMatching) [
            (input = (m_out = Bernoulli(0.5f0), m_in = NormalMeanVariance(1.0f0, 0.5f0) ), output = NormalWeightedMeanPrecision(4f0 * tiny, 4f0 * tiny) ), 
            (input = (m_out = Bernoulli(0.5f0), m_in = NormalMeanPrecision(1.0f0, 2.0f0) ), output = NormalWeightedMeanPrecision(4f0 * tiny, 4f0 * tiny) ), 
            (input = (m_out = Bernoulli(0.5f0), m_in = NormalWeightedMeanPrecision(2.0f0, 2.0f0) ), output = NormalWeightedMeanPrecision(4f0 * tiny, 4f0 * tiny) ),
        ]

        @test_rules [ with_float_conversions = false, atol = 1e-25 ] Probit(:in, MomentMatching) [
            (input = (m_out = Bernoulli(big"0.5"), m_in = NormalMeanVariance(big"1.0", big"0.5") ), output = NormalWeightedMeanPrecision(big"4.0" * tiny, big"4.0" * tiny) ), 
            (input = (m_out = Bernoulli(big"0.5"), m_in = NormalMeanPrecision(big"1.0", big"2.0") ), output = NormalWeightedMeanPrecision(big"4.0" * tiny, big"4.0" * tiny) ), 
            (input = (m_out = Bernoulli(big"0.5"), m_in = NormalWeightedMeanPrecision(big"2.0", big"2.0") ), output = NormalWeightedMeanPrecision(big"4.0" * tiny, big"4.0" * tiny) ),
        ]
        
    end 

end


end