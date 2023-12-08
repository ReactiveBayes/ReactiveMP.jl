module RulesContinuousTransitionTest

using Test, ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

import ReactiveMP: @test_rules, ctcompanion_matrix, getjacobians, getunits

@testset "rules:ContinuousTransition:y" begin
    @testset "Structured: (m_x::MultivariateNormalDistributionsFamily, q_a::MultivariateNormalDistributionsFamily, q_W::Any, meta::CTMeta)" begin
        # Example transformation function and vector length for CTMeta
        meta = CTMeta((a) -> reshape(a, 2, 3), 6)

        @test_rules [check_type_promotion = true] ContinuousTransition(:y, Marginalisation) [(
            input = (m_x = MvNormalMeanCovariance(zeros(3), diageye(3)), q_a = MvNormalMeanCovariance(zeros(6), diageye(6)), q_W = Wishart(3, diageye(2)), meta = meta),
            output = MvNormalMeanCovariance(zeros(2), 1 / 3 * diageye(2))
        )]
    end

    # Additional tests for edge cases, errors, or specific behaviors of the rule can be added here
end

end
