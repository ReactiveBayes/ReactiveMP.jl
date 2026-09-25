# The canary for the whole design: a variadic group, an indexed target `(:m, k)`, and
# declared dependencies. A group input is a tuple in member order with `nothing` where the
# dependency leaves a member out. Categorical inputs check promotion in Float32 and Float64
# only (see gamma_categorical_dirichlet_tests.jl).

@testitem "rules:NormalMixture:m" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions

    @test_message_update_rule(
        node = NormalMixture, target = (:m, 2), float_types = (Float32, Float64),
        cases = [(q = (out = PointMass(1.5), switch = Categorical([0.3, 0.7]), p = (nothing, GammaShapeRate(3.0, 1.0))),) => NormalMeanPrecision(1.5, 0.7 * 3)],
    )
    @test_throws ErrorException getresult(call_message_update_rule(NormalMixture, (:m, 1); q = (out = PointMass(1.5), switch = PointMass(1), p = (GammaShapeRate(3.0, 1.0), nothing))))
end

@testitem "rules:NormalMixture:p" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions

    # z = 0.3: shape 1 + z/2, rate z (var_out + var_m + (out - m)²) / 2 = 0.3 (0 + 1 + 1.5²) / 2
    @test_message_update_rule(
        node = NormalMixture, target = (:p, 1), float_types = (Float32, Float64),
        cases = [(q = (out = PointMass(1.5), switch = Categorical([0.3, 0.7]), m = (NormalMeanVariance(0.0, 1.0), nothing)),) => GammaShapeRate(1.15, 0.4875)],
    )
    @test_throws ErrorException getresult(call_message_update_rule(NormalMixture, (:p, 1); q = (out = PointMass(1.5), switch = PointMass(1), m = (NormalMeanVariance(0.0, 1.0), nothing))))
end

@testitem "rules:NormalMixture:switch-out-energy" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions
    using SpecialFunctions: digamma
    using StatsFuns: log2π

    # Each component's NormalMeanPrecision energy, E[τ] = 2 and E[log τ] = digamma(2).
    energy(m, v_m, out) = (log2π - digamma(2.0) + 2.0 * (v_m + abs2(m - out))) / 2
    U = [-energy(0.0, 1.0, 1.5), -energy(2.0, 1.0, 1.5)]
    weights = exp.(U .- maximum(U)) ./ sum(exp.(U .- maximum(U)))
    components = (m = (NormalMeanVariance(0.0, 1.0), NormalMeanVariance(2.0, 1.0)), p = (GammaShapeRate(2.0, 1.0), GammaShapeRate(2.0, 1.0)))

    @test_message_update_rule(
        node = NormalMixture, target = :switch, float_types = (Float32, Float64),
        cases = [(q = (out = PointMass(1.5), components...),) => Categorical(weights)],
    )
    # W = 0.3·2 + 0.7·4, ξ = 0.3·2·0 + 0.7·4·2
    @test_message_update_rule(
        node = NormalMixture, target = :out, float_types = (Float32, Float64),
        cases = [(q = (switch = Categorical([0.3, 0.7]), m = components.m, p = (GammaShapeRate(2.0, 1.0), GammaShapeRate(4.0, 1.0))),) => NormalWeightedMeanPrecision(5.6, 3.4)],
    )
    @test_average_energy(
        node = NormalMixture, float_types = (Float32, Float64),
        cases = [(q = (out = PointMass(1.5), switch = Categorical([0.3, 0.7]), components...),) => 0.3 * energy(0.0, 1.0, 1.5) + 0.7 * energy(2.0, 1.0, 1.5)],
    )
end

@testitem "rules:NormalMixture:multivariate" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    import ExponentialFamily: WishartFast

    # Wishart precisions and a two-dimensional `out`; the responsibilities are
    # clamped to [tiny, 1 - tiny], so a one-hot switch agrees to 1e-4.
    W1, W2 = Wishart(3.0, [2.0 -0.25; -0.25 1.0]), Wishart(3.0, [1.0 -0.25; -0.25 2.0])
    @test_message_update_rule(
        node = NormalMixture, target = (:m, 1), float_types = (Float32, Float64), atol = 1.0e-4,
        cases = [
            (q = (out = MvNormalWeightedMeanPrecision([6.75, 12.0], [4.5 -0.75; -0.75 4.5]), switch = Categorical([0.5, 0.5]), p = (W1, nothing)),) =>
                MvNormalMeanPrecision([2.0, 3.0], [3.0 -0.375; -0.375 1.5]),
            (q = (out = MvNormalMeanPrecision([3.75, 10.3125], [5.25 -0.75; -0.75 3.75]), switch = Categorical([0.75, 0.25]), p = (W1, nothing)),) =>
                MvNormalMeanPrecision([3.75, 10.3125], [4.5 -0.5625; -0.5625 2.25]),
            (q = (out = MvNormalMeanPrecision([0.75, 17.25], [3.0 -0.75; -0.75 6.0]), switch = Categorical([1.0, 0.0]), p = (W1, nothing)),) =>
                MvNormalMeanPrecision([0.75, 17.25], [6.0 -0.75; -0.75 3.0]),
        ],
    )
    I2 = [1.0 0.0; 0.0 1.0]
    points = (m = (PointMass([1.0, 0.0]), PointMass([-1.0, -2.0])), p = (PointMass(2.0 * I2), PointMass(3.0 * I2)))
    @test_message_update_rule(
        node = NormalMixture, target = :out, float_types = (Float32, Float64),
        cases = [
            (q = (switch = Categorical([0.5, 0.5]), points...),) => MvNormalWeightedMeanPrecision([-1 / 2, -3], [5 / 2 0; 0 5 / 2]),
            (q = (switch = Categorical([1.0, 0.0]), points...),) => MvNormalWeightedMeanPrecision([2.0, 0.0], [2.0 0.0; 0.0 2.0]),
            (q = (switch = Categorical([0.0, 1.0]), points...),) => MvNormalWeightedMeanPrecision([-3.0, -6.0], [3.0 0.0; 0.0 3.0]),
            (q = (switch = Categorical([0.5, 0.5]), m = (MvNormalMeanCovariance([2.0, 3.0], [2.0 0.0; 0.0 1.0]), MvNormalMeanPrecision([2.0, 3.0], [2.0 0.0; 0.0 1.0])), p = (W1, W2)),) =>
                MvNormalWeightedMeanPrecision([6.75, 12.0], [4.5 -0.75; -0.75 4.5]),
            (q = (switch = Categorical([0.75, 0.25]), m = (MvNormalWeightedMeanPrecision([2.0, 3.0], [2.0 0.0; 0.0 1.0]), MvNormalMeanPrecision([2.0, 3.0], [2.0 0.0; 0.0 1.0])), p = (W1, W2)),) =>
                MvNormalWeightedMeanPrecision([3.75, 10.3125], [5.25 -0.75; -0.75 3.75]),
            (q = (switch = Categorical([0.0, 1.0]), m = (MvNormalMeanCovariance([2.0, 3.0], [2.0 0.0; 0.0 1.0]), MvNormalWeightedMeanPrecision([2.0, 3.0], [2.0 0.0; 0.0 1.0])), p = (W1, W2)),) =>
                MvNormalWeightedMeanPrecision([0.75, 17.25], [3.0 -0.75; -0.75 6.0]),
        ],
    )
    @test_message_update_rule(
        node = NormalMixture, target = (:p, 1), float_types = (Float32, Float64), atol = 1.0e-4,
        cases = [
            (q = (out = MvNormalMeanPrecision([8.5], [0.5]), switch = Bernoulli(0.8), m = (MvNormalMeanPrecision([3.0], [0.1]), nothing)),) => WishartFast(2.2, fill(8.45, 1, 1)),
            (q = (out = MvNormalMeanCovariance([-3.0], [2.0]), switch = Bernoulli(0.5), m = (MvNormalMeanCovariance([5.0], [2.0]), nothing)),) => WishartFast(2.5, fill(34.0, 1, 1)),
            # z = 1/4, d = 2: 1 + z + d degrees of freedom, inverse scale z (I + I + ΔΔᵀ) for Δ = (1, 2).
            (q = (out = MvNormalMeanCovariance([1.0, 2.0], I2), switch = Categorical([0.25, 0.75]), m = (MvNormalMeanCovariance([0.0, 0.0], I2), nothing)),) =>
                WishartFast(3.25, [0.75 0.5; 0.5 1.5]),
        ],
    )
    @test_message_update_rule(
        node = NormalMixture, target = :switch, float_types = (Float32, Float64), atol = 1.0e-4,
        cases = [
            (q = (out = MvNormalMeanCovariance([8.5], [0.5]), m = (MvNormalMeanCovariance([5.0], [2.0]), MvNormalMeanCovariance([10.0], [3.0])), p = (Wishart(2.0, fill(0.25, 1, 1)), Wishart(4.0, fill(0.5, 1, 1)))),) =>
                Categorical([0.7713458788198754, 0.22865412118012463]),
        ],
    )
    # The energy is each component's MvNormalMeanPrecision energy, weighted by the switch.
    q_out, q_m, q_p = MvNormalMeanCovariance([0.0], [1.0]), (MvNormalMeanPrecision([1.0], [2.0]), MvNormalMeanPrecision([3.0], [4.0])), (WishartFast(3.0, fill(3.0, 1, 1)), WishartFast(4.0, fill(5.0, 1, 1)))
    component(k) = getresult(call_average_energy(MvNormalMeanPrecision; q = (out = q_out, μ = q_m[k], Λ = q_p[k])))
    @test getresult(call_average_energy(NormalMixture; q = (out = q_out, switch = Categorical([0.5, 0.5]), m = q_m, p = q_p))) ≈ (component(1) + component(2)) / 2
    @test GaussianMixture === NormalMixture
end
