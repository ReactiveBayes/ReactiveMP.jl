@testitem "rules:SoftDot:θ" tags = [:rules] begin
    using SoftDotMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    using LinearAlgebra: I

    # Semi-exhaustive combinations of input types, as v6's table, labelled y, x, γ by
    #     0: PointMass, 1: NormalMeanVariance, 2: MvNormalMeanCovariance, 3: GammaShapeScale.
    # Each output has weighted mean z = ⟨γ⟩ m_x m_y and precision D = ⟨γ⟩ (V_x + m_x m_xᵀ).
    @testset "VMP: Mean-field" begin
        q_x_mv = MvNormalMeanCovariance([5.0, 9.0], [11.0 13.0; 17.0 19.0])
        @test_message_update_rule(
            node = SoftDot, target = :θ, check_type_promotion = true,
            cases = [
                # 000
                (q = (y = PointMass(3.0), x = PointMass(5.0), γ = PointMass(2.0)),) => NormalWeightedMeanPrecision(30.0, 50.0),
                # 003
                (q = (y = PointMass(3.0), x = PointMass(5.0), γ = GammaShapeScale(2.0, 7.0)),) => NormalWeightedMeanPrecision(210.0, 350.0),
                # 010
                (q = (y = PointMass(3.0), x = NormalMeanVariance(2.0, 7.0), γ = PointMass(2.0)),) => NormalWeightedMeanPrecision(12.0, 22.0),
                # 013
                (q = (y = PointMass(3.0), x = NormalMeanVariance(2.0, 7.0), γ = GammaShapeScale(5.0, 11.0)),) => NormalWeightedMeanPrecision(330.0, 605.0),
                # 020
                (q = (y = PointMass(3.0), x = q_x_mv, γ = PointMass(2.0)),) => MvNormalWeightedMeanPrecision([30.0, 54.0], [72.0 116.0; 124.0 200.0]),
                # 023
                (q = (y = PointMass(3.0), x = q_x_mv, γ = GammaShapeScale(7.0, 23.0)),) => MvNormalWeightedMeanPrecision([2415.0, 4347.0], [5796.0 9338.0; 9982.0 16100.0]),
                # 100
                (q = (y = NormalMeanVariance(3.0, 7.0), x = PointMass(5.0), γ = PointMass(2.0)),) => NormalWeightedMeanPrecision(30.0, 50.0),
                # 103
                (q = (y = NormalMeanVariance(3.0, 7.0), x = PointMass(5.0), γ = GammaShapeScale(2.0, 7.0)),) => NormalWeightedMeanPrecision(210.0, 350.0),
                # 110
                (q = (y = NormalMeanVariance(3.0, 7.0), x = NormalMeanVariance(2.0, 7.0), γ = PointMass(2.0)),) => NormalWeightedMeanPrecision(12.0, 22.0),
                # 113
                (q = (y = NormalMeanVariance(3.0, 7.0), x = NormalMeanVariance(2.0, 7.0), γ = GammaShapeScale(5.0, 11.0)),) => NormalWeightedMeanPrecision(330.0, 605.0),
                # 120
                (q = (y = NormalMeanVariance(3.0, 7.0), x = q_x_mv, γ = PointMass(2.0)),) => MvNormalWeightedMeanPrecision([30.0, 54.0], [72.0 116.0; 124.0 200.0]),
                # 123
                (q = (y = NormalMeanVariance(3.0, 7.0), x = q_x_mv, γ = GammaShapeScale(7.0, 23.0)),) => MvNormalWeightedMeanPrecision([2415.0, 4347.0], [5796.0 9338.0; 9982.0 16100.0]),
            ],
        )
    end

    # TODO: these errors have to be caught in the implementations themselves. The error type and
    # message itself will not provide any information or might not match. The rule takes any
    # marginal, as v6's did, so it is found and fails inside, as in v6.
    @testset "VMP: Incorrect Inputs" begin
        # 2**: INCORRECT (y cannot be Mv)
        @test_throws MethodError call_message_update_rule(
            SoftDot, :θ; q = (y = MvNormalMeanCovariance([3.0, 7.0], [11.0, 13.0]), x = NormalMeanVariance(7.0, 11.0), γ = GammaShapeScale(13.0, 5.0)),
        )
        # NOTE: γ can theoretically be Any, so also NormalMeanVariance
    end

    @testset "Structured: (q_y_x::MultivariateNormalDistributionsFamily, q_γ::Any)" begin
        @test_message_update_rule(
            node = SoftDot, target = :θ, check_type_promotion = true,
            cases = [
                (clusters = ((:y, :x) => MvNormalMeanCovariance(ones(2), Matrix(1.0I, 2, 2)),), q = (γ = GammaShapeRate(1.0, 1.0),)) => NormalWeightedMeanPrecision(1.0, 2.0),
                (clusters = ((:y, :x) => MvNormalMeanCovariance(2 * ones(2), Matrix(1.0I, 2, 2)),), q = (γ = GammaShapeScale(2.0, 1.0),)) => NormalWeightedMeanPrecision(8.0, 10.0),
            ],
        )

        order = 2
        @test_message_update_rule(
            node = SoftDot, target = :θ, check_type_promotion = true,
            cases = [
                (clusters = ((:y, :x) => MvNormalMeanCovariance(ones(order + 1), Matrix(1.0I, order + 1, order + 1)),), q = (γ = GammaShapeRate(1.0, 1.0),)) =>
                    MvNormalWeightedMeanPrecision(ones(order), [2.0 1.0; 1.0 2.0]),
                (clusters = ((:y, :x) => MvNormalMeanCovariance(zeros(order + 1), Matrix(1.0I, order + 1, order + 1)),), q = (γ = GammaShapeRate(1.0, 1.0),)) =>
                    MvNormalWeightedMeanPrecision(zeros(order), [1.0 0.0; 0.0 1.0]),
            ],
        )
    end
end
