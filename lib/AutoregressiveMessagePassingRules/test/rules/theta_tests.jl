@testitem "rules:AR:θ" tags = [:rules] begin
    using AutoregressiveMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, LinearAlgebra

    diageye(n) = Matrix{Float64}(I, n, n)

    @testset "Mean-field: (q_y, q_x, q_γ), univariate" begin
        @test_message_update_rule(
            node = AR, target = :θ, algorithm = ARVMP(Univariate, 1, ARsafe()),
            cases = [
                (q = (y = NormalMeanVariance(1.0, 1.0), x = NormalMeanVariance(1.0, 1.0), γ = GammaShapeRate(1.0, 1.0)),) => NormalWeightedMeanPrecision(1.0, 2.0),
                (q = (y = NormalWeightedMeanPrecision(1.0, 1.0), x = NormalMeanPrecision(1.0, 2.0), γ = GammaShapeScale(2.0, 1.0)),) => NormalWeightedMeanPrecision(2.0, 3.0),
            ],
        )
    end

    @testset "Mean-field: (q_y, q_x, q_γ), multivariate" begin
        order = 2
        @test_message_update_rule(
            node = AR, target = :θ, algorithm = ARVMP(Multivariate, order, ARsafe()),
            cases = [
                (q = (y = MvNormalMeanCovariance(zeros(order), diageye(order)), x = MvNormalMeanCovariance(ones(order), diageye(order)), γ = GammaShapeRate(1.0, 1.0)),) =>
                    MvNormalWeightedMeanPrecision(zeros(2), [2.0 1.0; 1.0 2.0]),
                (q = (y = MvNormalMeanCovariance(ones(order), diageye(order)), x = MvNormalMeanCovariance(zeros(order), diageye(order)), γ = GammaShapeScale(1.0, 1.0)),) =>
                    MvNormalWeightedMeanPrecision([0.0, 0.0], [1.0 0.0; 0.0 1.0]),
                (q = (y = MvNormalMeanCovariance(ones(order), diageye(order)), x = MvNormalMeanCovariance(ones(order), diageye(order)), γ = Gamma(2.0, 1.0)),) =>
                    MvNormalWeightedMeanPrecision([2.0, 2.0], [4.0 2.0; 2.0 4.0]),
            ],
        )
    end

    @testset "Structured: (q_y_x, q_γ), univariate" begin
        @test_message_update_rule(
            node = AR, target = :θ, algorithm = ARVMP(Univariate, 1, ARsafe()),
            cases = [
                (clusters = ((:y, :x) => MvNormalMeanCovariance(ones(2), diageye(2)),), q = (γ = GammaShapeRate(1.0, 1.0),)) => NormalWeightedMeanPrecision(1.0, 2.0),
                (clusters = ((:y, :x) => MvNormalMeanCovariance(2 * ones(2), diageye(2)),), q = (γ = GammaShapeScale(2.0, 1.0),)) => NormalWeightedMeanPrecision(8.0, 10.0),
            ],
        )
    end

    @testset "Structured: (q_y_x, q_γ), multivariate" begin
        order = 2
        @test_message_update_rule(
            node = AR, target = :θ, algorithm = ARVMP(Multivariate, order, ARsafe()),
            cases = [
                (clusters = ((:y, :x) => MvNormalMeanCovariance(ones(2order), diageye(2order)),), q = (γ = GammaShapeRate(1.0, 1.0),)) =>
                    MvNormalWeightedMeanPrecision(ones(order), [2.0 1.0; 1.0 2.0]),
                (clusters = ((:y, :x) => MvNormalMeanCovariance(zeros(2order), diageye(2order)),), q = (γ = GammaShapeRate(1.0, 1.0),)) =>
                    MvNormalWeightedMeanPrecision(zeros(order), [1.0 0.0; 0.0 1.0]),
            ],
        )
    end
end
