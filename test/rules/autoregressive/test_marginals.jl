module RulesNormalAutoregressiveTest

using Test
using ReactiveMP
using Random
using LinearAlgebra
using Distributions

import ReactiveMP: @test_marginalrules

@testset "marginalrules:Autoregressive" begin
    @testset "y_x: (m_y::NormalDistributionsFamily, m_x::NormalDistributionsFamily, q_θ::NormalDistributionsFamily, q_γ::Any)" begin
        @test_marginalrules [check_type_promotion = true] Autoregressive(:y_x) [(
            input = (
                m_y = NormalMeanPrecision(0.0, 1.0),
                m_x = NormalMeanPrecision(0.0, 1.0),
                q_θ = NormalMeanPrecision(1.0, 1.0),
                q_γ = GammaShapeRate(1.0, 1.0),
                meta = ARMeta(Univariate, 1, ARsafe())
            ),
            output = MvNormalWeightedMeanPrecision(zeros(2), [2.0 -1.0; -1.0 3.0])
        )]
        # Test for multivariate m_x and m_y are absent due to the numerical instabilities induced by jitters (see marginal rule for autoregressive node)
    end
end
end
