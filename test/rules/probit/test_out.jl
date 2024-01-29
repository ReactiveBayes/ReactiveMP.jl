module RulesProbitOutTest

using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions
using StatsFuns: normcdf, normccdf, normlogcdf, normlogccdf, normlogpdf, normpdf

import ReactiveMP: @test_rules

@testitem "rules:Probit:out" begin
    @testset "Belief Propagation: (m_in::UnivariateNormalDistribution, )" begin
        @test_rules [check_type_promotion = true] Probit(:out, Marginalisation) [
            (input = (m_in = NormalMeanVariance(1, 0.5),), output = Bernoulli(normcdf(1 / sqrt(1 + 0.5)))),
            (input = (m_in = NormalMeanPrecision(1, 2),), output = Bernoulli(normcdf(1 / sqrt(1 + 0.5)))),
            (input = (m_in = NormalWeightedMeanPrecision(2, 2),), output = Bernoulli(normcdf(1 / sqrt(1 + 0.5)))),
            (input = (m_in = NormalMeanVariance(2, 0.25),), output = Bernoulli(normcdf(2 / sqrt(1 + 0.25)))),
            (input = (m_in = NormalMeanPrecision(2, 4),), output = Bernoulli(normcdf(2 / sqrt(1 + 0.25)))),
            (input = (m_in = NormalWeightedMeanPrecision(8, 4),), output = Bernoulli(normcdf(2 / sqrt(1 + 0.25))))
        ]
    end

    @testset "Belief Propagation: (m_in::PointMass, )" begin
        @test_rules [check_type_promotion = true] Probit(:out, Marginalisation) [
            (input = (m_in = PointMass(1),), output = Bernoulli(normcdf(1))),
            (input = (m_in = PointMass(2),), output = Bernoulli(normcdf(2))),
            (input = (m_in = PointMass(3),), output = Bernoulli(normcdf(3)))
        ]
    end
end

end
