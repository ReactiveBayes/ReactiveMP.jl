module RulesContinuousTransitionTest

using Test, ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

import ReactiveMP: @test_rules, ctcompanion_matrix, getmasks, getunits, WishartFast

@testset "rules:ContinuousTransition:W" begin
    @testset "Structured: (q_y_x::MultivariateNormalDistributionsFamily, q_a::MultivariateNormalDistributionsFamily, meta::CTMeta)" begin
        # Example transformation function and vector length for CTMeta
        meta = CTMeta((a) -> reshape(a, 2, 3), 6)

        @test_rules [check_type_promotion = true] ContinuousTransition(:W, Marginalisation) [
            (
                input = (
                    q_y_x = MvNormalMeanCovariance(zeros(5), diageye(5)),  # Adjust dimensions as needed
                    q_a = MvNormalMeanCovariance(zeros(6), diageye(6)),
                    meta = meta
                ),
                output = WishartFast(4, 4 * diageye(2))
            )
            # Additional test cases with different distributions and metadata settings
        ]
    end
    # Additional tests for edge cases, errors, or specific behaviors of the rule can be added here
end

end
