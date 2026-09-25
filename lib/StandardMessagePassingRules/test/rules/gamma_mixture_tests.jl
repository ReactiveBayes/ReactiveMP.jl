# GammaMixture: tables of cases, with a group as a tuple in member order and `nothing` where the
# dependency leaves a member out; its energy against the components' GammaShapeRate energies;
# and GammaShapeLikelihood, the shape message.

@testitem "rules:GammaMixture:a-b" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions

    γ(p, q_out, q_b) = p * (mean(log, q_out) + mean(log, q_b))
    @test_message_update_rule(
        node = GammaMixture, target = (:a, 1), float_types = (Float32, Float64),
        cases = [
            (q = (out = GammaShapeRate(2.0, 1.0), switch = Categorical([0.7, 0.3]), b = (GammaShapeRate(1.0, 2.0), nothing)),) =>
                GammaShapeLikelihood(0.7, γ(0.7, GammaShapeRate(2.0, 1.0), GammaShapeRate(1.0, 2.0))),
            (q = (out = GammaShapeRate(3.0, 2.0), switch = Bernoulli(0.5), b = (GammaShapeRate(1.0, 3.0), nothing)),) =>
                GammaShapeLikelihood(0.5, γ(0.5, GammaShapeRate(3.0, 2.0), GammaShapeRate(1.0, 3.0))),
        ],
    )
    # probvec(Bernoulli(0.8)) is (0.2, 0.8), so the first component's responsibility is 0.2.
    @test_message_update_rule(
        node = GammaMixture, target = (:b, 1), float_types = (Float32, Float64),
        cases = [
            (q = (out = GammaShapeRate(2.0, 1.0), switch = Categorical([0.6, 0.4]), a = (PointMass(1.0), nothing)),) => GammaShapeRate(1 + 0.6 * 1.0, 0.6 * 2.0),
            (q = (out = GammaShapeRate(4.0, 2.0), switch = Bernoulli(0.8), a = (GammaShapeRate(2.0, 3.0), nothing)),) => GammaShapeRate(1 + 0.2 * (2 / 3), 0.2 * 2.0),
        ],
    )
    @test_throws ErrorException getresult(call_message_update_rule(GammaMixture, (:b, 1); q = (out = GammaShapeRate(2.0, 1.0), switch = PointMass(1), a = (PointMass(1.0), nothing))))
    @test_throws ErrorException getresult(call_message_update_rule(GammaMixture, (:a, 1); q = (out = GammaShapeRate(2.0, 1.0), switch = PointMass(1), b = (GammaShapeRate(1.0, 2.0), nothing))))
end

@testitem "rules:GammaMixture:out-switch-energy" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions

    @test_message_update_rule(
        node = GammaMixture, target = :out, float_types = (Float32, Float64),
        cases = [
            (q = (switch = Categorical([0.7, 0.3]), a = (PointMass(2.0), PointMass(3.0)), b = (GammaShapeRate(1.0, 2.0), GammaShapeRate(2.0, 1.0))),) =>
                GammaShapeRate(0.7 * 2.0 + 0.3 * 3.0, 0.7 * 0.5 + 0.3 * 2.0),
            (q = (switch = Categorical([0.5, 0.5]), a = (GammaShapeRate(2.0, 1.0), GammaShapeRate(3.0, 1.0)), b = (GammaShapeRate(1.0, 3.0), GammaShapeRate(1.0, 2.0))),) =>
                GammaShapeRate(0.5 * 2.0 + 0.5 * 3.0, 0.5 / 3 + 0.5 / 2),
        ],
    )
    # ExponentialFamily's E[log Γ(a)] for a GammaShapeRate is a Float64 whatever its inputs
    # (ExponentialFamily.jl#322), so the switch rule and the energy are not checked in Float32;
    # a Categorical is not checked in BigFloat either, whose probabilities no longer sum to one
    # (see gamma_categorical_dirichlet_tests.jl).
    @test_message_update_rule(
        node = GammaMixture, target = :switch, float_types = (Float64,), atol = 1.0e-6,
        cases = [
            (q = (out = GammaShapeRate(2.0, 1.0), a = (GammaShapeRate(1.0, 2.0), GammaShapeRate(2.0, 3.0)), b = (GammaShapeRate(3.0, 1.0), GammaShapeRate(4.0, 2.0))),) =>
                Categorical([0.08088693183519022, 0.9191130681648099]),
        ],
    )
    # The components' GammaShapeRate energies, weighted by the switch.
    q_out, q_a, q_b = GammaShapeRate(1.0, 1.0), (GammaShapeRate(2.0, 3.0), GammaShapeRate(4.0, 5.0)), (GammaShapeRate(1.5, 2.5), GammaShapeRate(3.5, 4.5))
    component(k) = getresult(call_average_energy(GammaShapeRate; q = (out = q_out, α = q_a[k], β = q_b[k])))
    @test_average_energy(
        node = GammaMixture, float_types = (Float64,),
        cases = [(q = (out = q_out, switch = Categorical([0.2, 0.8]), a = q_a, b = q_b),) => 0.2 * component(1) + 0.8 * component(2)],
    )
end

@testitem "rules:GammaMixture:GammaShapeLikelihood" tags = [:rules] begin
    using StandardMessagePassingRules, ExponentialFamily, BayesBase, Distributions
    using SpecialFunctions: loggamma

    for p in (1.0, 2.0, 3.0), γ in (1.0, 2.0, 3.0), s in (1.0, 2.0)
        @test insupport(GammaShapeLikelihood(p, γ), s)
        @test !insupport(GammaShapeLikelihood(p, γ), -s)
    end
    d1, d2 = GammaShapeLikelihood(1.0, 2.0), GammaShapeLikelihood(2.0, 3.0)
    @test params(d1) == (1.0, 2.0)
    @test logpdf(d1, 1.0) ≈ 2.0 * 1.0 - 1.0 * loggamma(1.0)
    # Two shape likelihoods multiply into one, adding their parameters.
    @test BayesBase.default_prod_rule(GammaShapeLikelihood, GammaShapeLikelihood) == PreserveTypeProd(Distribution)
    product = prod(PreserveTypeProd(Distribution), d1, d2)
    @test product isa GammaShapeLikelihood && params(product) == (3.0, 5.0)
    @test minimum(support(d1)) == 0.0 && maximum(support(d1)) == Inf
end
