@testitem "rules:SoftDot:y" tags = [:rules] begin
    using SoftDotMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    using LinearAlgebra: I

    # Semi-exhaustive combinations of input types, as v6's table, labelled θ, x, γ by
    #     0: PointMass, 1: NormalMeanVariance, 2: MvNormalMeanCovariance, 3: a Gamma.
    # Each output is N(⟨θ⟩ᵀ⟨x⟩, ⟨γ⟩⁻¹), in its precision.
    @testset "VMP: Mean-field" begin
        q_θ_mv = MvNormalMeanCovariance([3.0, 7.0], [11.0, 13.0])
        q_x_mv = MvNormalMeanCovariance([17.0, 19.0], [23.0, 29.0])
        @test_message_update_rule(
            node = SoftDot, target = :y, check_type_promotion = true,
            cases = [
                # 000
                (q = (θ = PointMass(3.0), x = PointMass(11.0), γ = PointMass(7.0)),) => NormalMeanPrecision(33.0, 7.0),
                # 003
                (q = (θ = PointMass(3.0), x = PointMass(11.0), γ = GammaShapeRate(7.0, 5.0)),) => NormalMeanPrecision(33.0, 1.4),
                (q = (θ = PointMass(3.0), x = PointMass(11.0), γ = GammaShapeScale(7.0, 5.0)),) => NormalMeanPrecision(33.0, 35.0),
                # 110
                (q = (θ = NormalMeanVariance(3.0, 17.0), x = NormalMeanVariance(7.0, 11.0), γ = PointMass(13.0)),) => NormalMeanPrecision(21.0, 13.0),
                # 113
                (q = (θ = NormalMeanVariance(3.0, 17.0), x = NormalMeanVariance(7.0, 11.0), γ = GammaShapeRate(13.0, 5.0)),) => NormalMeanPrecision(21.0, 2.6),
                (q = (θ = NormalMeanVariance(3.0, 17.0), x = NormalMeanVariance(7.0, 11.0), γ = GammaShapeScale(13.0, 5.0)),) => NormalMeanPrecision(21.0, 65.0),
                # 220
                (q = (θ = q_θ_mv, x = q_x_mv, γ = PointMass(31.0)),) => NormalMeanPrecision(184.0, 31.0),
                # 223
                (q = (θ = q_θ_mv, x = q_x_mv, γ = GammaShapeRate(31.0, 5.0)),) => NormalMeanPrecision(184.0, 6.2),
                (q = (θ = q_θ_mv, x = q_x_mv, γ = GammaShapeScale(31.0, 5.0)),) => NormalMeanPrecision(184.0, 155.0),
            ],
        )
    end

    # TODO: these errors have to be caught in the implementations themselves. The error type and
    # message itself will not provide any information or might not match. The rule takes any
    # marginal, as v6's did, so it is found and fails inside, as in v6.
    @testset "VMP: Incorrect Inputs" begin
        to_y(q) = call_message_update_rule(SoftDot, :y; q)
        # 02*, 12*, 21*: INCORRECT (θ and x have to be of the same dimension)
        @test_throws MethodError to_y((θ = PointMass(7.0), x = MvNormalMeanCovariance([3.0, 7.0], [11.0, 13.0]), γ = GammaShapeRate(13.0, 5.0)))
        @test_throws MethodError to_y((θ = NormalMeanVariance(7.0, 11.0), x = MvNormalMeanCovariance([3.0, 7.0], [11.0, 13.0]), γ = GammaShapeRate(13.0, 5.0)))
        @test_throws MethodError to_y((θ = MvNormalMeanCovariance([3.0, 7.0], [11.0, 13.0]), x = NormalMeanVariance(7.0, 11.0), γ = GammaShapeRate(13.0, 5.0)))
    end

    @testset "VMP: structured rules" begin
        # From the message on `x`, AR's `y` message's first component (see the helpers' tests):
        # with D = W_x + ⟨γ⟩ V_θ, N(m_θᵀ D⁻¹ W_x m_x, m_θᵀ D⁻¹ m_θ + 1/⟨γ⟩).
        @test_message_update_rule(
            node = SoftDot, target = :y, check_type_promotion = true,
            cases = [
                (m = (x = NormalMeanVariance(1.0, 1.0),), q = (θ = NormalMeanVariance(1.0, 1.0), γ = GammaShapeRate(1.0, 1.0))) => NormalMeanVariance(0.5, 1.5),
                (m = (x = NormalWeightedMeanPrecision(1.0, 1.0),), q = (θ = NormalMeanPrecision(1.0, 2.0), γ = GammaShapeScale(2.0, 1.0))) => NormalMeanVariance(0.5, 1.0),
            ],
        )

        order = 2
        @test_message_update_rule(
            node = SoftDot, target = :y, check_type_promotion = true,
            cases = [
                (m = (x = MvNormalMeanCovariance(ones(order), Matrix(1.0I, order, order)),), q = (θ = MvNormalMeanCovariance(zeros(order), Matrix(1.0I, order, order)), γ = GammaShapeScale(1.0, 1.0))) =>
                    NormalMeanVariance(0.0, 1.0),
                (m = (x = MvNormalMeanCovariance(ones(order), Matrix(1.0I, order, order)),), q = (θ = MvNormalMeanCovariance(ones(order), Matrix(1.0I, order, order)), γ = Gamma(1.0, 1.0))) =>
                    NormalMeanVariance(1.0, 2.0),
            ],
        )
    end
end
