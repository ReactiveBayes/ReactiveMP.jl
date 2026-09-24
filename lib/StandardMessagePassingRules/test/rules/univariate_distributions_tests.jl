# Phase 5, step 3: the univariate distributions, against v6's own tables and node tests.

@testitem "rules:Beta" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    using SpecialFunctions: logbeta, digamma

    @test_message_update_rule(
        node = Beta, target = :out,
        cases = [
            (m = (a = PointMass(1.0), b = PointMass(2.0)),) => Beta(1.0, 2.0),
            (q = (a = PointMass(2.0), b = PointMass(2.0)),) => Beta(2.0, 2.0),
        ],
    )
    @test_marginal_update_rule(
        node = Beta, target = (:out, :a, :b),
        cases = [
            (m = (out = Beta(1.0, 2.0), a = PointMass(1.0), b = PointMass(2.0)),) =>
                FactorizedCluster((:out,) => Beta(1.0, 3.0), (:a,) => PointMass(1.0), (:b,) => PointMass(2.0)),
        ],
    )
    # E[log x] = digamma(α) - digamma(α + β), E[log(1 - x)] = digamma(β) - digamma(α + β).
    @test_average_energy(
        node = Beta,
        cases = [
            (q = (out = Beta(2.0, 3.0), a = PointMass(1.5), b = PointMass(2.5)),) =>
                logbeta(1.5, 2.5) - 0.5 * (digamma(2.0) - digamma(5.0)) - 1.5 * (digamma(3.0) - digamma(5.0)),
        ],
    )
end

@testitem "rules:Bernoulli" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions

    @test_message_update_rule(
        node = Bernoulli, target = :out,
        cases = [
            (m = (p = PointMass(0.2),),) => Bernoulli(0.2),
            (q = (p = PointMass(0.3),),) => Bernoulli(0.3),
            (q = (p = Beta(0.2, 0.2),),) => Bernoulli(0.5),
            (m = (p = Beta(2.0, 6.0),),) => ExpectedWithAnnotations(Bernoulli(0.25); logscale = 0),
        ],
    )
    @test_message_update_rule(
        node = Bernoulli, target = :p,
        cases = [
            (m = (out = PointMass(0.2),),) => ExpectedWithAnnotations(Beta(12 / 10, 9 / 5); logscale = -log(2)),
            (q = (out = PointMass(1.0),),) => ExpectedWithAnnotations(Beta(2.0, 1.0); logscale = -log(2)),
            (q = (out = Bernoulli(0.3),),) => Beta(13 / 10, 17 / 10),
        ],
    )
    # BigFloat(0.7) + BigFloat(0.3) is not one, so Categorical cannot be promoted there.
    @test_message_update_rule(
        node = Bernoulli, target = :p, float_types = (Float32, Float64),
        cases = [(q = (out = Categorical([0.7, 0.3]),),) => Beta(13 / 10, 17 / 10)],
    )
    @test_throws ArgumentError call_message_update_rule(Bernoulli, :p; q = (out = Categorical([0.2, 0.3, 0.5]),))
    @test_marginal_update_rule(
        node = Bernoulli, target = (:out, :p),
        cases = [
            (m = (out = PointMass(1.0), p = Beta(2.0, 1.0)),) => FactorizedCluster((:out,) => PointMass(1.0), (:p,) => Beta(3.0, 1.0)),
            (m = (out = Bernoulli(0.8), p = PointMass(1.0)),) => FactorizedCluster((:out,) => Bernoulli(1.0), (:p,) => PointMass(1.0)),
        ],
    )
    @test_average_energy(
        node = Bernoulli,
        cases = [(q = (out = Bernoulli(0.3), p = PointMass(0.6)),) => -0.3 * log(0.6) - 0.7 * log(0.4)],
    )
end

@testitem "rules:Gamma-GammaInverse" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    using SpecialFunctions: digamma, loggamma

    @test_message_update_rule(
        node = Gamma, target = :out,
        cases = [
            (m = (α = PointMass(2.0), θ = PointMass(3.0)),) => Gamma(2.0, 3.0),
            (q = (α = PointMass(2.0), θ = PointMass(0.5)),) => Gamma(2.0, 0.5),
        ],
    )
    # Gamma(2, scale 1) times Gamma(2, scale 1) is Gamma(3, scale 1/2).
    @test_marginal_update_rule(
        node = Gamma, target = (:out, :α, :θ),
        cases = [
            (m = (out = Gamma(2.0, 1.0), α = PointMass(2.0), θ = PointMass(1.0)),) =>
                FactorizedCluster((:out,) => Gamma(3.0, 0.5), (:α,) => PointMass(2.0), (:θ,) => PointMass(1.0)),
        ],
    )
    # The energy's E[x/θ] is E[x]·E[1/θ], here 6 · 1/4 for out ~ Gamma(3, scale 2) and
    # θ ~ Gamma(3, scale 2), where v6 took E[x]/E[θ] = 1. E[log x] = digamma(3) + log(2).
    @test_average_energy(
        node = Gamma,
        cases = [
            (q = (out = Gamma(3.0, 2.0), α = PointMass(2.0), θ = Gamma(3.0, 2.0)),) =>
                loggamma(2.0) + 2 * (digamma(3.0) + log(2.0)) - (2 - 1) * (digamma(3.0) + log(2.0)) + 6 * (1 / 4),
            (q = (out = Gamma(3.0, 2.0), α = PointMass(2.0), θ = PointMass(0.5)),) =>
                loggamma(2.0) + 2 * log(0.5) - (2 - 1) * (digamma(3.0) + log(2.0)) + 6 / 0.5,
        ],
    )
    # A Gamma q_α, E[α] = 2: E[loggamma(α)] has no closed form, and is ExponentialFamily's, which
    # is Float64 only (ExponentialFamily.jl#322).
    q_α = GammaShapeRate(4.0, 2.0)
    @test call_average_energy(Gamma; q = (out = Gamma(3.0, 2.0), α = q_α, θ = PointMass(0.5))) ≈
        mean(loggamma, q_α) + 2 * log(0.5) - (2 - 1) * (digamma(3.0) + log(2.0)) + 6 / 0.5
    @test_message_update_rule(
        node = GammaInverse, target = :out,
        cases = [
            (m = (α = PointMass(1.0), θ = PointMass(2.0)),) => GammaInverse(1.0, 2.0),
            (q = (α = PointMass(1.0), θ = PointMass(2.0)),) => GammaInverse(1.0, 2.0),
        ],
    )
    @test_marginal_update_rule(
        node = GammaInverse, target = (:out, :α, :θ),
        cases = [
            (m = (out = GammaInverse(1.0, 2.0), α = PointMass(1.0), θ = PointMass(2.0)),) =>
                FactorizedCluster((:out,) => GammaInverse(3.0, 4.0), (:α,) => PointMass(1.0), (:θ,) => PointMass(2.0)),
        ],
    )
    # By hand: -α log θ + loggamma(α) + (α + 1) E[log x] + θ E[1/x], with, for x ~ InvGamma(a, b),
    # E[log x] = log(b) - digamma(a) and E[1/x] = a/b. v6's node tests expected θ/E[x] in the
    # last term, so their values (-0.268…, -1.434…) are not reproduced.
    inverse_gamma_energy(α, θ, a, b) = -α * log(θ) + loggamma(α) + (α + 1) * (log(b) - digamma(a)) + θ * a / b
    @test call_average_energy(GammaInverse; q = (out = GammaInverse(2.0, 1.0), α = PointMass(2.0), θ = PointMass(1.0))) ≈ inverse_gamma_energy(2.0, 1.0, 2.0, 1.0)
    @test call_average_energy(GammaInverse; q = (out = GammaInverse(42.0, 42.0), α = PointMass(42.0), θ = PointMass(42.0))) ≈ inverse_gamma_energy(42.0, 42.0, 42.0, 42.0)
    @test inverse_gamma_energy(2.0, 1.0, 2.0, 1.0) ≈ 0.7316469947045985 rtol = 1.0e-9
end

@testitem "rules:HalfNormal" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions

    @test_message_update_rule(
        node = HalfNormal, target = :out,
        cases = [
            (q = (v = PointMass(1.0),),) => Truncated(Normal(0.0, 1.0), 0.0, Inf),
            (q = (v = PointMass(100.0),),) => Truncated(Normal(0.0, 10.0), 0.0, Inf),
        ],
    )
    # v6's node tests.
    @test call_average_energy(HalfNormal; q = (out = GammaShapeRate(2.0, 1.0), v = PointMass(2.0))) ≈ 2.072364942925
    @test call_average_energy(HalfNormal; q = (out = GammaInverse(3.0, 1.0), v = PointMass(2.0))) ≈ 0.6973649429247
end

@testitem "rules:Poisson" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions

    @test_message_update_rule(
        node = Poisson, target = :out,
        cases = [(m = (l = PointMass(0.2),),) => Poisson(0.2)],
    )
    @test_message_update_rule(
        node = Poisson, target = :out, atol = 1.0e-4,
        cases = [
            (q = (l = Gamma(1.0, 1.0),),) => Poisson(0.56146),
            (q = (l = GammaShapeRate(1.0, 0.5),),) => Poisson(1.12292),
        ],
    )
    @test_message_update_rule(
        node = Poisson, target = :l,
        cases = [
            (m = (out = PointMass(1.0),),) => Gamma(2.0, 1.0),
            (q = (out = Poisson(0.3),),) => Gamma(1.3, 1.0),
        ],
    )
    @test_marginal_update_rule(
        node = Poisson, target = (:out, :l),
        cases = [(m = (out = PointMass(1.0), l = Gamma(2.0, 1.0)),) => FactorizedCluster((:out,) => PointMass(1.0), (:l,) => Gamma(3.0, 0.5))],
    )
    # v6's node tests: point masses give -log p(k | λ), and a Poisson q_out its entropy.
    @test all(isapprox(call_average_energy(Poisson; q = (out = PointMass(k), l = PointMass(l))), -logpdf(Poisson(l), k); rtol = 1.0e-12) for l in 1:20, k in 1:20)
    @test all(isapprox(call_average_energy(Poisson; q = (out = Poisson(k), l = PointMass(k))), entropy(Poisson(k)); rtol = 1.0e-3) for k in 1:100)
end

@testitem "rules:Uniform" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions

    for (a, b) in ((:m, :m), (:q, :m), (:m, :q), (:q, :q))
        inputs = a === b ? NamedTuple{(a,)}(((a = PointMass(1.0), b = PointMass(2.0)),)) :
            NamedTuple{(a, b)}(((a = PointMass(1.0),), (b = PointMass(2.0),)))
        @test call_message_update_rule(Uniform, :out; inputs...) == Uniform(1.0, 2.0)
    end
    @test_message_update_rule(node = Uniform, target = :out, cases = [(m = (a = PointMass(2.0), b = PointMass(3.0)),) => Uniform(2.0, 3.0)])

    @test call_average_energy(Uniform; q = (out = Beta(0.3, 0.7), a = PointMass(0.0), b = PointMass(1.0))) == 0.0
    @test BayesBase.default_prod_rule(Uniform, Beta) == PreserveTypeProd(Distribution)
    @test prod(PreserveTypeProd(Distribution), Uniform(0.0, 1.0), Beta(2.0, 5.0)) === Beta(2.0, 5.0)
    @test_throws ArgumentError prod(PreserveTypeProd(Distribution), Uniform(0.0, 2.0), Beta(2.0, 5.0))
end

@testitem "rules:Uninformative" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    using BayesBase: TerminalProdArgument

    @test call_message_update_rule(Uninformative, :out) === Uninformative()
    @test call_average_energy(Uninformative; q = (out = NormalMeanVariance(0.0, 1.0),)) === 0.0

    # The product must not be affected (v6's node tests).
    @test prod(GenericProd(), Uninformative(), NormalMeanVariance(0, 1)) == NormalMeanVariance(0, 1)
    @test prod(GenericProd(), NormalMeanVariance(3, 4), Uninformative()) == NormalMeanVariance(3, 4)
    @test prod(GenericProd(), Uninformative(), Uninformative()) === Uninformative()
    @test prod(GenericProd(), Uninformative(), TerminalProdArgument(NormalMeanVariance(0, 1))) == TerminalProdArgument(NormalMeanVariance(0, 1))
    @test prod(GenericProd(), TerminalProdArgument(PointMass(0)), Uninformative()) === TerminalProdArgument(PointMass(0))
end
