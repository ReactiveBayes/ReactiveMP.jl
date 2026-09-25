@testitem "rules:AR:y" tags = [:rules] begin
    using AutoregressiveMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, LinearAlgebra
    using AutoregressiveMessagePassingRules: ARTransitionMatrix

    diageye(n) = Matrix{Float64}(I, n, n)

    @testset "Mean-field: (q_x, q_θ, q_γ), univariate" begin
        @test_message_update_rule(
            node = AR, target = :y, algorithm = ARVMP(Univariate, 1, ARsafe()),
            cases = [
                (q = (x = NormalMeanVariance(1.0, 1.0), θ = NormalMeanVariance(1.0, 1.0), γ = GammaShapeRate(1.0, 1.0)),) => NormalMeanVariance(1.0, 1.0),
                (q = (x = NormalWeightedMeanPrecision(1.0, 1.0), θ = NormalMeanPrecision(1.0, 2.0), γ = GammaShapeScale(2.0, 1.0)),) => NormalMeanVariance(1.0, 0.5),
            ],
        )
    end

    @testset "Mean-field: (q_x, q_θ, q_γ), multivariate" begin
        order = 2
        @test_message_update_rule(
            node = AR, target = :y, algorithm = ARVMP(Multivariate, order, ARsafe()),
            cases = [
                (q = (x = MvNormalMeanCovariance(zeros(order), diageye(order)), θ = MvNormalMeanCovariance(ones(order), diageye(order)), γ = GammaShapeRate(1.0, 1.0)),) =>
                    MvNormalMeanCovariance(zeros(2), ARTransitionMatrix(order, 1.0)),
                (q = (x = MvNormalMeanCovariance(ones(order), diageye(order)), θ = MvNormalMeanCovariance(zeros(order), diageye(order)), γ = GammaShapeScale(1.0, 1.0)),) =>
                    MvNormalMeanCovariance([0.0, 1.0], ARTransitionMatrix(order, 1.0)),
                (q = (x = MvNormalMeanCovariance(ones(order), diageye(order)), θ = MvNormalMeanCovariance(ones(order), diageye(order)), γ = Gamma(2.0, 1.0)),) =>
                    MvNormalMeanCovariance([2.0, 1.0], ARTransitionMatrix(order, 2.0)),
            ],
        )
    end

    @testset "Structured: (m_x, q_θ, q_γ), univariate" begin
        algorithm = ARVMP(Univariate, 1, ARsafe())
        @test_message_update_rule(
            node = AR, target = :y, algorithm = algorithm,
            cases = [
                (m = (x = NormalMeanVariance(1.0, 1.0),), q = (θ = NormalMeanVariance(1.0, 1.0), γ = GammaShapeRate(1.0, 1.0))) => NormalMeanVariance(0.5, 1.5),
                (m = (x = NormalWeightedMeanPrecision(1.0, 1.0),), q = (θ = NormalMeanPrecision(1.0, 2.0), γ = GammaShapeScale(2.0, 1.0))) => NormalMeanVariance(0.5, 1.0),
            ],
        )
        # A multivariate message under a univariate algorithm, and a univariate one under a
        # multivariate algorithm, have no meaning.
        @test_throws MethodError getresult(
            call_message_update_rule(
                AR, :y; m = (x = MvNormalMeanPrecision([0.0], [1.0;;]),), q = (θ = NormalMeanPrecision(1.0, 1.0), γ = GammaShapeRate(1.0, 1.0)), algorithm,
            )
        )
        @test_throws MethodError getresult(
            call_message_update_rule(
                AR, :y; m = (x = NormalMeanPrecision(0.0, 1.0),), q = (θ = NormalMeanPrecision(1.0, 1.0), γ = GammaShapeRate(1.0, 1.0)),
                algorithm = ARVMP(Multivariate, 1, ARsafe()),
            )
        )
    end

    @testset "Structured: (m_x, q_θ, q_γ), multivariate" begin
        order = 2
        @test_message_update_rule(
            node = AR, target = :y, algorithm = ARVMP(Multivariate, order, ARsafe()),
            cases = [
                (m = (x = MvNormalMeanCovariance(zeros(order), diageye(order)),), q = (θ = MvNormalMeanCovariance(ones(order), diageye(order)), γ = GammaShapeRate(1.0, 1.0))) =>
                    MvNormalMeanCovariance([0.0, 0.0], [2.0 0.5; 0.5 0.5]),
                (m = (x = MvNormalMeanCovariance(ones(order), diageye(order)),), q = (θ = MvNormalMeanCovariance(zeros(order), diageye(order)), γ = GammaShapeScale(1.0, 1.0))) =>
                    MvNormalMeanCovariance([0.0, 0.5], [1.0 0.0; 0.0 0.5]),
                (m = (x = MvNormalMeanCovariance(ones(order), diageye(order)),), q = (θ = MvNormalMeanCovariance(ones(order), diageye(order)), γ = Gamma(1.0, 1.0))) =>
                    MvNormalMeanCovariance([1.0, 0.5], [2.0 0.5; 0.5 0.5]),
            ],
        )
    end

    @testset "No rule under the default algorithm" begin
        @test_throws MessagePassingRulesBase.RuleNotFoundError getresult(
            call_message_update_rule(
                AR, :y; q = (x = NormalMeanVariance(1.0, 1.0), θ = NormalMeanVariance(1.0, 1.0), γ = GammaShapeRate(1.0, 1.0)),
            )
        )
    end
end
