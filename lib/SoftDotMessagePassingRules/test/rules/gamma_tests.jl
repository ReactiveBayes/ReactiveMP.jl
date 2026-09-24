@testitem "rules:SoftDot:γ" tags = [:rules] begin
    using SoftDotMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    using LinearAlgebra: I

    # Semi-exhaustive combinations of input types, as v6's table, labelled y, θ, x by
    #     0: PointMass, 1: NormalMeanVariance, 2: MvNormalMeanCovariance.
    # Each output is Γ(3/2, B/2) with
    #     B = V_y + m_y² - 2 m_y m_θᵀm_x + tr(V_x V_θ) + m_θᵀ(V_x + m_x m_xᵀ)m_θ + m_xᵀV_θ m_x.
    @testset "VMP: Mean-field" begin
        @test_message_update_rule(
            node = SoftDot, target = :γ, check_type_promotion = true,
            cases = [
                # 000
                (q = (y = PointMass(3.0), θ = PointMass(5.0), x = PointMass(2.0)),) => GammaShapeRate(3 / 2, 49 / 2),
                # 001
                (q = (y = PointMass(3.0), θ = PointMass(5.0), x = NormalMeanVariance(2.0, 7.0)),) => GammaShapeRate(3 / 2, 224 / 2),
                # 010
                (q = (y = PointMass(3.0), θ = NormalMeanVariance(2.0, 7.0), x = PointMass(5.0)),) => GammaShapeRate(3 / 2, 224 / 2),
                # 011
                (q = (y = PointMass(3.0), θ = NormalMeanVariance(2.0, 7.0), x = NormalMeanVariance(5.0, 11.0)),) => GammaShapeRate(3 / 2, 345 / 2),
                # 100
                (q = (y = NormalMeanVariance(2.0, 3.0), θ = PointMass(5.0), x = PointMass(7.0)),) => GammaShapeRate(3 / 2, 1092 / 2),
                # 101
                (q = (y = NormalMeanVariance(3.0, 7.0), θ = PointMass(2.0), x = NormalMeanVariance(5.0, 9.0)),) => GammaShapeRate(3 / 2, 92 / 2),
                # 110
                (q = (y = NormalMeanVariance(3.0, 7.0), θ = NormalMeanVariance(5.0, 9.0), x = PointMass(2.0)),) => GammaShapeRate(3 / 2, 92 / 2),
                # 111
                (q = (y = NormalMeanVariance(3.0, 7.0), θ = NormalMeanVariance(5.0, 9.0), x = NormalMeanVariance(11.0, 13.0)),) => GammaShapeRate(3 / 2, 4242 / 2),
                # 122
                (
                    q = (
                        y = NormalMeanVariance(3.0, 7.0),
                        θ = MvNormalMeanCovariance([5.0, 9.0], [11.0 13.0; 17.0 19.0]),
                        x = MvNormalMeanCovariance([23.0, 29.0], [31.0 37.0; 41.0 43.0]),
                    ),
                ) => GammaShapeRate(3 / 2, 191032 / 2),
            ],
        )
    end

    # TODO: these errors have to be caught in the implementations themselves. The error type and
    # message itself will not provide any information or might not match. The rules take any
    # marginal, as v6's did, so a rule is found and fails inside, as in v6.
    @testset "VMP: Incorrect Inputs" begin
        to_γ(q) = call_message_update_rule(SoftDot, :γ; q)
        # 2**: INCORRECT (y cannot be Mv)
        @test_throws DimensionMismatch to_γ((y = MvNormalMeanCovariance([3.0, 7.0], [11.0, 13.0]), θ = NormalMeanVariance(7.0, 11.0), x = NormalMeanVariance(13.0, 5.0)))
        # *02, *20, *12, *21: INCORRECT (θ and x have to have the same dimensions)
        @test_throws MethodError to_γ((y = NormalMeanVariance(3.0, 7.0), θ = PointMass(7.0), x = MvNormalMeanCovariance([3.0, 7.0], [11.0, 13.0])))
        @test_throws MethodError to_γ((y = NormalMeanVariance(3.0, 7.0), θ = MvNormalMeanCovariance([3.0, 7.0], [11.0, 13.0]), x = PointMass(7.0)))
        @test_throws MethodError to_γ((y = NormalMeanVariance(3.0, 7.0), θ = NormalMeanVariance(7.0, 11.0), x = MvNormalMeanCovariance([3.0, 7.0], [11.0, 13.0])))
        @test_throws MethodError to_γ((y = NormalMeanVariance(3.0, 7.0), θ = MvNormalMeanCovariance([3.0, 7.0], [11.0, 13.0]), x = NormalMeanVariance(7.0, 11.0)))
    end

    @testset "Structured: (q_y_x::MultivariateNormalDistributionsFamily, q_θ::Any)" begin
        @test_message_update_rule(
            node = SoftDot, target = :γ, check_type_promotion = true,
            cases = [
                (clusters = ((:y, :x) => MvNormalMeanCovariance(ones(2), Matrix(1.0I, 2, 2)),), q = (θ = NormalMeanPrecision(1.0, 1.0),)) => GammaShapeRate(3 / 2, 2.0),
                (clusters = ((:y, :x) => MvNormalMeanCovariance(2 * ones(2), Matrix(1.0I, 2, 2)),), q = (θ = NormalMeanPrecision(2.0, 1.0),)) => GammaShapeRate(3 / 2, 7.0),
            ],
        )

        order = 2
        @test_message_update_rule(
            node = SoftDot, target = :γ, check_type_promotion = true,
            cases = [
                (
                    clusters = ((:y, :x) => MvNormalMeanCovariance(ones(order + 1), Matrix(1.0I, order + 1, order + 1)),),
                    q = (θ = MvNormalMeanPrecision(ones(order), Matrix(1.0I, order, order)),),
                ) => GammaShapeRate(3 / 2, 4.0),
                (
                    clusters = ((:y, :x) => MvNormalMeanCovariance(ones(order + 1), Matrix(1.0I, order + 1, order + 1)),),
                    q = (θ = MvNormalMeanPrecision(zeros(order), Matrix(1.0I, order, order)),),
                ) => GammaShapeRate(3 / 2, 3.0),
            ],
        )
    end
end
