@testitem "marginalrules:SoftDot" tags = [:rules] begin
    using SoftDotMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    using LinearAlgebra: I

    # The joint q(y, x): the messages' natural parameters, and the factor's precision
    # ⟨γ⟩ [1 -⟨θ⟩ᵀ; -⟨θ⟩ ⟨θθᵀ⟩] added to theirs.
    @testset "y_x: (m_y::UnivariateNormalDistributionsFamily, m_x::UnivariateNormalDistributionsFamily, q_θ::UnivariateNormalDistributionsFamily, q_γ::Any)" begin
        @test_marginal_update_rule(
            node = SoftDot, target = (:y, :x), check_type_promotion = true,
            cases = [
                (m = (y = NormalMeanPrecision(0.0, 1.0), x = NormalMeanPrecision(0.0, 1.0)), q = (θ = NormalMeanPrecision(1.0, 1.0), γ = GammaShapeRate(1.0, 1.0))) =>
                    MvNormalWeightedMeanPrecision(zeros(2), [2.0 -1.0; -1.0 3.0]),
            ],
        )
    end

    @testset "y_x: (m_y::UnivariateNormalDistributionsFamily, m_x::MultivariateNormalDistributionsFamily, q_θ::MultivariateNormalDistributionsFamily, q_γ::Any)" begin
        order = 2
        @test_marginal_update_rule(
            node = SoftDot, target = (:y, :x), check_type_promotion = true,
            cases = [
                (
                    m = (y = NormalMeanPrecision(1.0, 1.0), x = MvNormalMeanCovariance(ones(order), Matrix(1.0I, order, order))),
                    q = (θ = MvNormalMeanCovariance(ones(order), Matrix(1.0I, order, order)), γ = GammaShapeRate(1.0, 1.0)),
                ) => MvNormalWeightedMeanPrecision(ones(3), [2.0 -1.0 -1.0; -1.0 3.0 1.0; -1.0 1.0 3.0]),
            ],
        )
    end
end
