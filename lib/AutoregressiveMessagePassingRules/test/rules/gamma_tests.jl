@testitem "rules:AR:γ" tags = [:rules] begin
    using AutoregressiveMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, LinearAlgebra

    diageye(n) = Matrix{Float64}(I, n, n)

    @testset "Mean-field: (q_y, q_x, q_θ), univariate" begin
        @test_message_update_rule(
            node = AR, target = :γ, algorithm = ARVMP(Univariate, 1, ARsafe()),
            cases = [
                (q = (y = NormalMeanVariance(1.0, 1.0), x = NormalMeanVariance(1.0, 1.0), θ = NormalMeanPrecision(1.0, 1.0)),) => GammaShapeRate(3 / 2, 3 / 2),
                (q = (y = NormalWeightedMeanPrecision(1.0, 1.0), x = NormalMeanPrecision(1.0, 2.0), θ = NormalMeanPrecision(2.0, 1.0)),) => GammaShapeRate(3 / 2, 5 / 2),
            ],
        )
    end

    @testset "Mean-field: (q_y, q_x, q_θ), multivariate" begin
        order = 2
        @test_message_update_rule(
            node = AR, target = :γ, algorithm = ARVMP(Multivariate, order, ARsafe()),
            cases = [
                (q = (y = MvNormalMeanCovariance(zeros(order), diageye(order)), x = MvNormalMeanCovariance(ones(order), diageye(order)), θ = MvNormalMeanCovariance(ones(order), diageye(order))),) =>
                    GammaShapeRate(3 / 2, 9 / 2),
                (q = (y = MvNormalMeanCovariance(ones(order), diageye(order)), x = MvNormalMeanCovariance(zeros(order), diageye(order)), θ = MvNormalMeanCovariance(ones(order), diageye(order))),) =>
                    GammaShapeRate(3 / 2, 2.0),
                (q = (y = MvNormalMeanCovariance(ones(order), diageye(order)), x = MvNormalMeanCovariance(ones(order), diageye(order)), θ = MvNormalMeanCovariance(ones(order), diageye(order))),) =>
                    GammaShapeRate(3 / 2, 3.0),
            ],
        )
    end

    @testset "Structured: (q_y_x, q_θ), univariate" begin
        @test_message_update_rule(
            node = AR, target = :γ, algorithm = ARVMP(Univariate, 1, ARsafe()),
            cases = [
                (clusters = ((:y, :x) => MvNormalMeanCovariance(ones(2), diageye(2)),), q = (θ = NormalMeanPrecision(1.0, 1.0),)) => GammaShapeRate(3 / 2, 2.0),
                (clusters = ((:y, :x) => MvNormalMeanCovariance(2 * ones(2), diageye(2)),), q = (θ = NormalMeanPrecision(2.0, 1.0),)) => GammaShapeRate(3 / 2, 7.0),
            ],
        )
    end

    @testset "Structured: (q_y_x, q_θ), multivariate" begin
        order = 2
        @test_message_update_rule(
            node = AR, target = :γ, algorithm = ARVMP(Multivariate, order, ARsafe()),
            cases = [
                (clusters = ((:y, :x) => MvNormalMeanCovariance(ones(2order), diageye(2order)),), q = (θ = MvNormalMeanPrecision(ones(order), diageye(order)),)) =>
                    GammaShapeRate(3 / 2, 4.0),
                (clusters = ((:y, :x) => MvNormalMeanCovariance(ones(2order), diageye(2order)),), q = (θ = MvNormalMeanPrecision(zeros(order), diageye(order)),)) =>
                    GammaShapeRate(3 / 2, 3.0),
            ],
        )
    end
end
