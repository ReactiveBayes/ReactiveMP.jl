module RulesBIFMHelperInTest

using ReactiveMP, Random, ExponentialFamily, BayesBase

import ReactiveMP: @test_rules

@testitem "rules:BIFMHelper:in" begin
    @testset "Belief Propagation: (m_out::Any, )" begin
        @test_rules [check_type_promotion = true] BIFMHelper(:in, Marginalisation) [
            (input = (m_out = PointMass(1.0),), output = PointMass(1.0)),
            (input = (m_out = NormalMeanVariance(9.5, 3.2),), output = NormalMeanVariance(9.5, 3.2)),
            (input = (m_out = MvNormalWeightedMeanPrecision([1.0, 2.0], [3.0 0; 0 2.0]),), output = MvNormalWeightedMeanPrecision([1.0, 2.0], [3.0 0; 0 2.0]))
        ]
    end
end

end
