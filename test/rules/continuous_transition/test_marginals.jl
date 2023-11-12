module RulesContinuousTransitionTest

using Test, ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions, LinearAlgebra
import ReactiveMP: @test_marginalrules

@testset "marginalrules:ContinuousTransition" begin
    @testset "y_x: (m_y::NormalDistributionsFamily, m_x::NormalDistributionsFamily, q_a::NormalDistributionsFamily, q_W::Any)" begin
        meta = CTMeta((a) -> reshape(a, 2, 3), 6)

        @test_marginalrules [check_type_promotion = true] ContinuousTransition(:y_x) [(
            input = (
                m_y = MvNormalMeanPrecision(ones(2), diageye(2)),
                m_x = MvNormalMeanPrecision(ones(3), diageye(3)),
                q_a = MvNormalMeanPrecision(ones(6), diageye(6)),
                q_W = Wishart(2, diageye(2)),
                meta = meta
            ),
            output = MvNormalWeightedMeanPrecision(zeros(5), diageye(5))
        )]
    end
end
end
