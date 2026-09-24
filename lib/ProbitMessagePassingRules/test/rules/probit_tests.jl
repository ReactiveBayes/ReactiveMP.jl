# Probit, from v6's `test/rules/probit/` and `test/nodes/predefined/probit_tests.jl`. The rules
# towards `out` exist under both algorithms; the log-density messages towards `in` are compared
# at points, as TestUtils compares no log-density by value.

@testitem "rules:Probit:out" tags = [:rules] begin
    using ProbitMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    using StatsFuns: normcdf

    for algorithm in (ProbitEP(), DefaultAlgorithm())
        @test_message_update_rule(
            node = Probit, target = :out, algorithm = algorithm,
            cases = [
                (m = (in = NormalMeanVariance(1, 0.5),),) => Bernoulli(normcdf(1 / sqrt(1 + 0.5))),
                (m = (in = NormalMeanPrecision(1, 2),),) => Bernoulli(normcdf(1 / sqrt(1 + 0.5))),
                (m = (in = NormalWeightedMeanPrecision(2, 2),),) => Bernoulli(normcdf(1 / sqrt(1 + 0.5))),
                (m = (in = NormalMeanVariance(2, 0.25),),) => Bernoulli(normcdf(2 / sqrt(1 + 0.25))),
                (m = (in = NormalMeanPrecision(2, 4),),) => Bernoulli(normcdf(2 / sqrt(1 + 0.25))),
                (m = (in = NormalWeightedMeanPrecision(8, 4),),) => Bernoulli(normcdf(2 / sqrt(1 + 0.25))),
                (m = (in = PointMass(1),),) => Bernoulli(normcdf(1)),
                (m = (in = PointMass(2),),) => Bernoulli(normcdf(2)),
                (m = (in = PointMass(3),),) => Bernoulli(normcdf(3)),
            ],
        )
    end
    # Under mean-field, a point-mass `q(in)` gives the same.
    @test_message_update_rule(
        node = Probit, target = :out, algorithm = DefaultAlgorithm(),
        cases = [(q = (in = PointMass(1),),) => Bernoulli(normcdf(1)), (q = (in = PointMass(2),),) => Bernoulli(normcdf(2))],
    )
end

@testitem "rules:Probit:in, expectation propagation" tags = [:rules] begin
    using ProbitMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    using BayesBase: tiny

    positive = NormalWeightedMeanPrecision(0.6723616582693972, 0.32950039939606945)
    negative = NormalWeightedMeanPrecision(-0.821224653874111, 0.7003447377360019)
    cavities = (NormalMeanVariance(1.0, 0.5), NormalMeanPrecision(1.0, 2.0), NormalWeightedMeanPrecision(2.0, 2.0))
    @test_message_update_rule(
        node = Probit, target = :in,
        cases = vcat(
            [(m = (out = PointMass(1.0), in = c),) => positive for c in cavities],
            [(m = (out = PointMass(0.0), in = c),) => negative for c in cavities],
            [(m = (out = Bernoulli(1.0), in = c),) => positive for c in cavities],
            [(m = (out = Bernoulli(0.8), in = c),) => NormalWeightedMeanPrecision(0.427017495944859, 0.199141999223396) for c in cavities],
            [(m = (out = Bernoulli(0.0), in = c),) => negative for c in cavities],
            # An uninformative output leaves the smallest precision the rule allows.
            [(m = (out = Bernoulli(0.5), in = c),) => NormalWeightedMeanPrecision(0.0, 1.0 * tiny) for c in cavities],
        ),
    )
    @test_throws ArgumentError call_message_update_rule(Probit, :in; m = (out = PointMass(2.0), in = NormalMeanVariance(1.0, 0.5)))
end

@testitem "rules:Probit:in, belief propagation" tags = [:rules] begin
    using ProbitMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    using StatsFuns: normcdf

    # log(1 - p + (2p - 1) Φ(z)), under DefaultAlgorithm, from a message or a point-mass marginal.
    for (p, expected) in ((1.0, z -> log(normcdf(z))), (0.8, z -> log(0.2 + 0.6 * normcdf(z))), (0.5, z -> log(0.5)), (0.0, z -> log(1 - normcdf(z))))
        from_message = call_message_update_rule(Probit, :in; m = (out = PointMass(p),), algorithm = DefaultAlgorithm())
        from_marginal = call_message_update_rule(Probit, :in; q = (out = PointMass(p),), algorithm = DefaultAlgorithm())
        for z in (-2.0, -0.5, 0.0, 0.7, 3.0)
            @test logpdf(from_message, z) ≈ expected(z) atol = 1.0e-5
            @test logpdf(from_marginal, z) ≈ expected(z) atol = 1.0e-5
        end
    end
end

@testitem "rules:Probit:marginals" tags = [:rules] begin
    using ProbitMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    using StatsFuns: normpdf, normcdf

    # The joint of a point-mass output keeps the output, and gives `in` the tilted distribution,
    # which is the cavity times the message towards `in`.
    for (p, m_in) in ((1.0, NormalMeanVariance(1.0, 0.5)), (0.0, NormalMeanPrecision(-0.5, 3.0)), (1.0, NormalWeightedMeanPrecision(2.0, 2.0)))
        joint = call_marginal_update_rule(Probit, (:out, :in); m = (out = PointMass(p), in = m_in))
        message = call_message_update_rule(Probit, :in; m = (out = PointMass(p), in = m_in))
        @test joint[(:out,)] == PointMass(p)
        @test all(mean_var(joint[(:in,)]) .≈ mean_var(prod(GenericProd(), m_in, message)))
    end
    # For y = 1 the tilted distribution of N(μ, v) under Φ has the closed form, with
    # γ = μ / √(1 + v) and r = φ(γ) / Φ(γ): mean μ + v r / √(1 + v), variance v - v² r (γ + r) / (1 + v).
    μ, v = 1.0, 0.5
    γ = μ / sqrt(1 + v)
    r = normpdf(γ) / normcdf(γ)
    @test_marginal_update_rule(
        node = Probit, target = (:out, :in),
        cases = [
            (m = (out = PointMass(1.0), in = NormalMeanVariance(μ, v)),) =>
                FactorizedCluster((:out,) => PointMass(1.0), (:in,) => NormalMeanVariance(μ + v * r / sqrt(1 + v), v - v^2 * r * (γ + r) / (1 + v))),
        ],
    )
end

@testitem "rules:Probit:energy" tags = [:rules] begin
    using ProbitMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions

    # E[-log Φ(x)] = E[-log Φ(-x)] = 1 for a standard normal x, whatever the output's probability.
    for algorithm in (ProbitEP(), ProbitEP(p = 100), DefaultAlgorithm())
        @test call_average_energy(Probit; q = (out = Bernoulli(1), in = NormalMeanVariance(0.0, 1.0)), algorithm) ≈ 1.0
        @test call_average_energy(Probit; q = (out = PointMass(1), in = NormalMeanVariance(0.0, 1.0)), algorithm) ≈ 1.0
    end
    for k in 0:0.1:1
        @test call_average_energy(Probit; q = (out = Bernoulli(k), in = NormalMeanVariance(0.0, 1.0)), algorithm = ProbitEP(p = 100)) ≈ 1.0
    end
    # A wide q(in) and 100 points reach x ≈ ±37, where v6's log(Φ(x)) underflowed to -Inf: the
    # energy was Inf, or NaN for a point-mass output (0 ⋅ -Inf). log Φ is computed directly now.
    for q_out in (PointMass(1.0), PointMass(0.0), Bernoulli(0.3))
        energy = call_average_energy(Probit; q = (out = q_out, in = NormalMeanVariance(-2.0, 4.0)), algorithm = ProbitEP(p = 100))
        @test isfinite(energy) && energy > 0
    end
end

@testitem "rules:Probit:node" tags = [:rules] begin
    using ProbitMessagePassingRules, MessagePassingRulesBase, ExponentialFamily
    using MessagePassingRulesBase: initial_messages, default_algorithm, dependencies_spec, target_dependencies, Target

    @test default_algorithm(Probit) === ProbitEP(32)
    @test initial_messages(Probit) == (:in => NormalMeanPrecision(0.0, 100.0),)
    spec = dependencies_spec(Probit, ProbitEP())
    @test map(d -> (d.container, d.key), target_dependencies(spec, Target(:in))) == ((:m, :out), (:m, :in))
    @test map(d -> (d.container, d.key), target_dependencies(spec, Target(:out))) == ((:m, :in),)
end
