@testitem "rules:SoftDot:x" tags = [:rules] begin
    using SoftDotMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    using LinearAlgebra: I

    # Semi-exhaustive combinations of input types, labelled y, θ, γ by
    #     0: PointMass, 1: NormalMeanVariance, 2: MvNormalMeanCovariance, 3: GammaShapeScale.
    # Each output has weighted mean z = ⟨γ⟩ m_θ m_y and precision D = ⟨γ⟩ (V_θ + m_θ m_θᵀ).
    @testset "VMP: Mean-field" begin
        q_θ_mv = MvNormalMeanCovariance([5.0, 9.0], [11.0 13.0; 17.0 19.0])
        @test_message_update_rule(
            node = SoftDot, target = :x, check_type_promotion = true,
            cases = [
                # 000
                (q = (y = PointMass(3.0), θ = PointMass(5.0), γ = PointMass(2.0)),) => NormalWeightedMeanPrecision(30.0, 50.0),
                # 003
                (q = (y = PointMass(3.0), θ = PointMass(5.0), γ = GammaShapeScale(2.0, 7.0)),) => NormalWeightedMeanPrecision(210.0, 350.0),
                # 010
                (q = (y = PointMass(3.0), θ = NormalMeanVariance(2.0, 7.0), γ = PointMass(2.0)),) => NormalWeightedMeanPrecision(12.0, 22.0),
                # 013
                (q = (y = PointMass(3.0), θ = NormalMeanVariance(2.0, 7.0), γ = GammaShapeScale(5.0, 11.0)),) => NormalWeightedMeanPrecision(330.0, 605.0),
                # 020
                (q = (y = PointMass(3.0), θ = q_θ_mv, γ = PointMass(2.0)),) => MvNormalWeightedMeanPrecision([30.0, 54.0], [72.0 116.0; 124.0 200.0]),
                # 023
                (q = (y = PointMass(3.0), θ = q_θ_mv, γ = GammaShapeScale(7.0, 23.0)),) => MvNormalWeightedMeanPrecision([2415.0, 4347.0], [5796.0 9338.0; 9982.0 16100.0]),
                # 100
                (q = (y = NormalMeanVariance(3.0, 7.0), θ = PointMass(5.0), γ = PointMass(2.0)),) => NormalWeightedMeanPrecision(30.0, 50.0),
                # 103
                (q = (y = NormalMeanVariance(3.0, 7.0), θ = PointMass(5.0), γ = GammaShapeScale(2.0, 7.0)),) => NormalWeightedMeanPrecision(210.0, 350.0),
                # 110
                (q = (y = NormalMeanVariance(3.0, 7.0), θ = NormalMeanVariance(2.0, 7.0), γ = PointMass(2.0)),) => NormalWeightedMeanPrecision(12.0, 22.0),
                # 113
                (q = (y = NormalMeanVariance(3.0, 7.0), θ = NormalMeanVariance(2.0, 7.0), γ = GammaShapeScale(5.0, 11.0)),) => NormalWeightedMeanPrecision(330.0, 605.0),
                # 120
                (q = (y = NormalMeanVariance(3.0, 7.0), θ = q_θ_mv, γ = PointMass(2.0)),) => MvNormalWeightedMeanPrecision([30.0, 54.0], [72.0 116.0; 124.0 200.0]),
                # 123
                (q = (y = NormalMeanVariance(3.0, 7.0), θ = q_θ_mv, γ = GammaShapeScale(7.0, 23.0)),) => MvNormalWeightedMeanPrecision([2415.0, 4347.0], [5796.0 9338.0; 9982.0 16100.0]),
            ],
        )
    end

    # TODO: these errors have to be caught in the implementations themselves. The error type and
    # message itself will not provide any information or might not match. The rule takes any
    # marginal, so it is found and fails inside.
    @testset "VMP: Incorrect Inputs" begin
        # 2**: INCORRECT (y cannot be Mv)
        @test_throws MethodError call_message_update_rule(
            SoftDot, :x; q = (y = MvNormalMeanCovariance([3.0, 7.0], [11.0, 13.0]), θ = NormalMeanVariance(7.0, 11.0), γ = GammaShapeScale(13.0, 5.0)),
        )
        # NOTE: γ can theoretically be Any, so also NormalMeanVariance
    end

    @testset "VMP: structured rules" begin
        # From the message on `y`: C = m_θ / (V_y + 1/⟨γ⟩), weighted mean C m_y, precision
        # C m_θᵀ + ⟨γ⟩ V_θ.
        @test_message_update_rule(
            node = SoftDot, target = :x, check_type_promotion = true,
            cases = [
                (m = (y = NormalMeanVariance(1.0, 1.0),), q = (θ = NormalMeanVariance(1.0, 1.0), γ = GammaShapeRate(1.0, 1.0))) => NormalWeightedMeanPrecision(0.5, 1.5),
                (m = (y = NormalWeightedMeanPrecision(1.0, 1.0),), q = (θ = NormalMeanPrecision(1.0, 2.0), γ = GammaShapeScale(1.0, 1.0))) => NormalWeightedMeanPrecision(0.5, 1.0),
            ],
        )

        order = 2
        @test_message_update_rule(
            node = SoftDot, target = :x, check_type_promotion = false,
            cases = [
                (m = (y = NormalMeanVariance(0.0, 1.0),), q = (θ = MvNormalMeanCovariance(ones(order), Matrix(1.0I, order, order)), γ = GammaShapeRate(1.0, 1.0))) =>
                    MvNormalWeightedMeanPrecision([0.0, 0.0], [1.5 0.5; 0.5 1.5]),
                (m = (y = NormalMeanVariance(1.0, 1.0),), q = (θ = MvNormalMeanCovariance(zeros(order), Matrix(1.0I, order, order)), γ = GammaShapeScale(1.0, 1.0))) =>
                    MvNormalWeightedMeanPrecision([0.0, 0.0], [1.0 0.0; 0.0 1.0]),
                (m = (y = NormalMeanVariance(1.0, 1.0),), q = (θ = MvNormalMeanCovariance(ones(order), Matrix(1.0I, order, order)), γ = Gamma(1.0, 1.0))) =>
                    MvNormalWeightedMeanPrecision([0.5, 0.5], [1.5 0.5; 0.5 1.5]),
            ],
        )
    end
end
