module RulesContinuousTransitionTestA

using Test, ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

import ReactiveMP: @test_rules, getmasks, getunits

@testset "rules:ContinuousTransition:a" begin
    @testset "Structured: (q_y_x::MultivariateNormalDistributionsFamily, q_a::NormalDistributionsFamily, q_W::Any, meta::CTMeta)" begin
        # Example transformation function and vector length for CTMeta
        meta = CTMeta((a) -> reshape(a, 2, 3), 6)

        @test_rules [check_type_promotion = true] ContinuousTransition(:a, Marginalisation) [(
            input = (
                q_y_x = MvNormalMeanCovariance([zeros(2); zeros(3)], [diageye(2) zeros(2, 3); zeros(3, 2) diageye(3)]),
                q_a = MvNormalMeanCovariance(zeros(6), diageye(6)),
                q_W = Wishart(3, diageye(2)),
                meta = meta
            ),
            output = MvNormalWeightedMeanPrecision(zeros(6), 3 * diageye(6))
        )]
    end
end

end
