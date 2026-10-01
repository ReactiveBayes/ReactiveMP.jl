@testitem "rules:AR:γ" tags = [:rules] begin
    using AutoregressiveMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, LinearAlgebra, StableRNGs

    diageye(n) = Matrix{Float64}(I, n, n)

    @testset "Mean-field: (q_y, q_x, q_θ), univariate" begin
        @test_message_update_rule(
            node = AR, target = :γ, algorithm = ARVMP(Univariate, 1, ARsafe()),
            cases = [
                (q = (y = NormalMeanVariance(1.0, 1.0), x = NormalMeanVariance(1.0, 1.0), θ = NormalMeanPrecision(1.0, 1.0)),) => GammaShapeRate(3 / 2, 2.0),
                (q = (y = NormalWeightedMeanPrecision(1.0, 1.0), x = NormalMeanPrecision(1.0, 2.0), θ = NormalMeanPrecision(2.0, 1.0)),) => GammaShapeRate(3 / 2, 11 / 4),
            ],
        )
    end

    @testset "Mean-field: (q_y, q_x, q_θ), multivariate" begin
        order = 2
        @test_message_update_rule(
            node = AR, target = :γ, algorithm = ARVMP(Multivariate, order, ARsafe()),
            cases = [
                (q = (y = MvNormalMeanCovariance(zeros(order), diageye(order)), x = MvNormalMeanCovariance(ones(order), diageye(order)), θ = MvNormalMeanCovariance(ones(order), diageye(order))),) =>
                    GammaShapeRate(3 / 2, 11 / 2),
                (q = (y = MvNormalMeanCovariance(ones(order), diageye(order)), x = MvNormalMeanCovariance(zeros(order), diageye(order)), θ = MvNormalMeanCovariance(ones(order), diageye(order))),) =>
                    GammaShapeRate(3 / 2, 3.0),
                (q = (y = MvNormalMeanCovariance(ones(order), diageye(order)), x = MvNormalMeanCovariance(ones(order), diageye(order)), θ = MvNormalMeanCovariance(ones(order), diageye(order))),) =>
                    GammaShapeRate(3 / 2, 4.0),
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

    # The average energy is (-E[log γ] + log 2π + E[γ] B) / 2 for both factorisations, and the
    # message towards γ is Gamma(3/2, B/2) with the same B: E[(y₁ - θᵀx)²]. Two gammas of one
    # shape give B from the energy.
    @testset "the rate is half the average energy's expected squared residual" begin
        function energy_residual(algorithm, inputs)
            a, b1, b2 = 2.0, 1.0, 3.0
            e(b) = getresult(call_average_energy(AR; merge(inputs, (q = merge(inputs.q, (γ = GammaShapeRate(a, b),)),))..., algorithm))
            return (2 * (e(b1) - e(b2)) - log(b1 / b2)) / (a * (1 / b1 - 1 / b2))
        end
        rng = StableRNG(42)
        spd(n) = (M = randn(rng, n, n); M * M' / n + I)
        for order in (1, 2, 3)
            algorithm = order == 1 ? ARVMP(Univariate, 1, ARsafe()) : ARVMP(Multivariate, order, ARsafe())
            gaussian(n) = n == 1 && order == 1 ? NormalMeanVariance(randn(rng), 1 + rand(rng)) : MvNormalMeanCovariance(randn(rng, n), spd(n))
            q_θ = gaussian(order)
            meanfield = (q = (y = gaussian(order), x = gaussian(order), θ = q_θ),)
            structured = (clusters = ((:y, :x) => MvNormalMeanCovariance(randn(rng, 2order), spd(2order)),), q = (θ = q_θ,))
            for inputs in (meanfield, structured)
                message = getresult(call_message_update_rule(AR, :γ; inputs..., algorithm))
                @test shape(message) == 3 / 2
                @test 2 * rate(message) ≈ energy_residual(algorithm, inputs) rtol = 1.0e-10
            end
        end
    end
end
