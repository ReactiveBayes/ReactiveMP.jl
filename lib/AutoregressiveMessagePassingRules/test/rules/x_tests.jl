@testitem "rules:AR:x" tags = [:rules] begin
    using AutoregressiveMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, LinearAlgebra

    diageye(n) = Matrix{Float64}(I, n, n)

    @testset "Mean-field: (q_y, q_θ, q_γ), univariate" begin
        @test_message_update_rule(
            node = AR, target = :x, algorithm = ARVMP(Univariate, 1, ARsafe()),
            cases = [
                (q = (y = NormalMeanVariance(1.0, 1.0), θ = NormalMeanVariance(1.0, 1.0), γ = GammaShapeRate(1.0, 1.0)),) => NormalWeightedMeanPrecision(1.0, 2.0),
                (q = (y = NormalWeightedMeanPrecision(1.0, 1.0), θ = NormalMeanPrecision(1.0, 2.0), γ = GammaShapeScale(2.0, 1.0)),) => NormalWeightedMeanPrecision(2.0, 3.0),
            ],
        )
    end

    @testset "Mean-field: (q_y, q_θ, q_γ), multivariate" begin
        order = 2
        @test_message_update_rule(
            node = AR, target = :x, algorithm = ARVMP(Multivariate, order, ARsafe()),
            cases = [
                (q = (y = MvNormalMeanCovariance(zeros(order), diageye(order)), θ = MvNormalMeanCovariance(ones(order), diageye(order)), γ = GammaShapeRate(1.0, 1.0)),) =>
                    MvNormalWeightedMeanPrecision(zeros(2), [2.0 1.0; 1.0 2.0]),
                (q = (y = MvNormalMeanCovariance(ones(order), diageye(order)), θ = MvNormalMeanCovariance(zeros(order), diageye(order)), γ = GammaShapeScale(1.0, 1.0)),) =>
                    MvNormalWeightedMeanPrecision([0.0, 0.0], [1.0 0.0; 0.0 1.0]),
                (q = (y = MvNormalMeanCovariance(ones(order), diageye(order)), θ = MvNormalMeanCovariance(ones(order), diageye(order)), γ = Gamma(2.0, 1.0)),) =>
                    MvNormalWeightedMeanPrecision([2.0, 2.0], [4.0 2.0; 2.0 4.0]),
            ],
        )
    end

    @testset "Structured: (m_y, q_θ, q_γ), univariate" begin
        algorithm = ARVMP(Univariate, 1, ARsafe())
        @test_message_update_rule(
            node = AR, target = :x, algorithm = algorithm,
            cases = [
                (m = (y = NormalMeanVariance(1.0, 1.0),), q = (θ = NormalMeanVariance(1.0, 1.0), γ = GammaShapeRate(1.0, 1.0))) => NormalWeightedMeanPrecision(0.5, 1.5),
                (m = (y = NormalWeightedMeanPrecision(1.0, 1.0),), q = (θ = NormalMeanPrecision(1.0, 2.0), γ = GammaShapeScale(1.0, 1.0))) => NormalWeightedMeanPrecision(0.5, 1.0),
            ],
        )
        # A multivariate message under a univariate algorithm, and a univariate one under a
        # multivariate algorithm, have no meaning.
        @test_throws MethodError call_message_update_rule(
            AR, :x; m = (y = MvNormalMeanPrecision([0.0], [1.0;;]),), q = (θ = NormalMeanPrecision(1.0, 1.0), γ = GammaShapeRate(1.0, 1.0)), algorithm,
        )
        @test_throws MethodError call_message_update_rule(
            AR, :x; m = (y = NormalMeanPrecision(0.0, 1.0),), q = (θ = NormalMeanPrecision(1.0, 1.0), γ = GammaShapeRate(1.0, 1.0)),
            algorithm = ARVMP(Multivariate, 1, ARsafe()),
        )
    end

    @testset "Structured: (m_y, q_θ, q_γ), multivariate" begin
        order = 2
        @test_message_update_rule(
            node = AR, target = :x, algorithm = ARVMP(Multivariate, order, ARsafe()),
            cases = [
                (m = (y = MvNormalMeanCovariance(zeros(order), diageye(order)),), q = (θ = MvNormalMeanCovariance(ones(order), diageye(order)), γ = GammaShapeRate(1.0, 1.0))) =>
                    MvNormalWeightedMeanPrecision([0.0, 0.0], [2.5 0.5; 0.5 1.5]),
                (m = (y = MvNormalMeanCovariance(ones(order), diageye(order)),), q = (θ = MvNormalMeanCovariance(zeros(order), diageye(order)), γ = GammaShapeScale(1.0, 1.0))) =>
                    MvNormalWeightedMeanPrecision([1.0, 0.0], [2.0 0.0; 0.0 1.0]),
                (m = (y = MvNormalMeanCovariance(ones(order), diageye(order)),), q = (θ = MvNormalMeanCovariance(ones(order), diageye(order)), γ = Gamma(1.0, 1.0))) =>
                    MvNormalWeightedMeanPrecision([1.5, 0.5], [2.5 0.5; 0.5 1.5]),
            ],
        )
    end
end
