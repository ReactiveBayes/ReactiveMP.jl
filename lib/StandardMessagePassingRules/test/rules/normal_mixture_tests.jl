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
    @test_throws ErrorException call_message_update_rule(NormalMixture, (:m, 1); q = (out = PointMass(1.5), switch = PointMass(1), p = (GammaShapeRate(3.0, 1.0), nothing)))
end

@testitem "rules:NormalMixture:p" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions

    # z = 0.3: shape 1 + z/2, rate z (var_out + var_m + (out - m)²) / 2 = 0.3 (0 + 1 + 1.5²) / 2
    @test_message_update_rule(
        node = NormalMixture, target = (:p, 1), float_types = (Float32, Float64),
        cases = [(q = (out = PointMass(1.5), switch = Categorical([0.3, 0.7]), m = (NormalMeanVariance(0.0, 1.0), nothing)),) => GammaShapeRate(1.15, 0.4875)],
    )
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
