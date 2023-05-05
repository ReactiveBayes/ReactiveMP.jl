module RulesContinuousTransitionTest

using Test
using ReactiveMP
using Random
using LinearAlgebra
using Distributions

import ReactiveMP: @test_marginalrules

# @call_marginalrule ContinuousTransition(:y_x) (m_y = MvNormalMeanPrecision(ones(2), diageye(2)), m_x = MvNormalMeanPrecision(ones(2), diageye(2)), q_h = MvNormalMeanPrecision(ones(4), diageye(4)), q_Λ = Wishart(2, diageye(2)), meta = CTMeta(2, 2))
# @call_marginalrule ContinuousTransition(:y_x) (m_y = MvNormalMeanPrecision(ones(2), diageye(2)), m_x = MvNormalMeanPrecision(ones(3), diageye(3)), q_h = MvNormalMeanPrecision(ones(6), diageye(6)), q_Λ = Wishart(2, diageye(2)), meta = CTMeta(2, 3))

@testset "marginalrules:ContinuousTransition" begin
    @testset "y_x: (m_y::NormalDistributionsFamily, m_x::NormalDistributionsFamily, q_θ::NormalDistributionsFamily, q_γ::Any)" begin
        @test_marginalrules [with_float_conversions = true] ContinuousTransition(:y_x) [(
            input = (
                m_y = MvNormalMeanPrecision(ones(2), diageye(2)),
                m_x = MvNormalMeanPrecision(ones(2), diageye(2)),
                q_h = MvNormalMeanPrecision(ones(4), diageye(4)),
                q_Λ = Wishart(2, diageye(2)),
                meta = CTMeta(2, 2)
            ),
            output = MvNormalWeightedMeanPrecision(zeros(2), [2.0 -1.0; -1.0 3.0])
        )]
    end
end
end
