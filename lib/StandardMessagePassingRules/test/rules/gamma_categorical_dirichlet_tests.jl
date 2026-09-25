# E[log x] under Gamma(a, rate b) is digamma(a) - log(b); under Dirichlet(a) it is
# digamma(aᵢ) - digamma(Σa).

@testitem "rules:GammaShapeRate" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions
    using SpecialFunctions: digamma, loggamma

    @test_message_update_rule(
        node = GammaShapeRate, target = :out,
        cases = [
            (m = (α = PointMass(2.0), β = PointMass(3.0)),) => GammaShapeRate(2.0, 3.0),
            (q = (α = PointMass(2.0), β = GammaShapeRate(2.0, 4.0)),) => GammaShapeRate(2.0, 0.5),
        ],
    )
    @test_average_energy(
        node = GammaShapeRate,
        cases = [
            (q = (out = GammaShapeRate(2.0, 3.0), α = PointMass(2.0), β = PointMass(3.0)),) =>
                loggamma(2.0) - 2 * log(3.0) - (2 - 1) * (digamma(2.0) - log(3.0)) + 3 * (2 / 3),
        ],
    )
    @verify_message_update_rule(node = GammaShapeRate, target = :out, q = (α = PointMass(2.0), β = GammaShapeRate(2.0, 4.0)))
end

@testitem "rules:Categorical" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    using SpecialFunctions: digamma

    softened = (a -> (ρ = exp.(digamma.(a) .- digamma(sum(a))); ρ ./ sum(ρ)))([1.0, 3.0])
    # Promotion is checked in Float32 and Float64 only, here and below: converting a
    # Categorical whose Float64 probabilities sum to one only approximately into BigFloat
    # fails Distributions' probability-vector check, before any rule is involved.
    @test_message_update_rule(
        node = Categorical, target = :out, float_types = (Float32, Float64),
        cases = [
            (m = (p = Dirichlet([1.0, 3.0]),),) => ExpectedWithLogScale(Categorical([0.25, 0.75]), 0),
            (q = (p = Dirichlet([1.0, 3.0]),),) => Categorical(softened),
            (m = (p = PointMass([0.2, 0.8]),),) => Categorical([0.2, 0.8]),
            (q = (p = PointMass([0.2, 0.8]),),) => Categorical([0.2, 0.8]),
        ],
    )
    @test_message_update_rule(
        node = Categorical, target = :p, float_types = (Float32, Float64),
        cases = [
            (q = (out = Categorical([0.3, 0.7]),),) => ExpectedWithLogScale(Dirichlet([1.3, 1.7]), -log(2.0)),
            (q = (out = PointMass([0.0, 1.0]),),) => ExpectedWithLogScale(Dirichlet([1.0, 2.0]), -log(2.0)),
        ],
    )
    @test_throws ArgumentError getresult(call_message_update_rule(Categorical, :p; q = (out = PointMass([0.5, 0.5]),)))
    @test_average_energy(
        node = Categorical, float_types = (Float32, Float64),
        cases = [(q = (out = Categorical([0.3, 0.7]), p = Dirichlet([1.0, 3.0])),) => -sum([0.3, 0.7] .* (digamma.([1.0, 3.0]) .- digamma(4.0)))],
    )
end

@testitem "rules:Dirichlet" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions
    using SpecialFunctions: digamma, loggamma

    @test_message_update_rule(
        node = Dirichlet, target = :out,
        cases = [
            (m = (a = PointMass([1.0, 2.0]),),) => Dirichlet([1.0, 2.0]),
            (q = (a = PointMass([1.0, 2.0]),),) => Dirichlet([1.0, 2.0]),
        ],
    )
    @test_average_energy(
        node = Dirichlet,
        cases = [
            (q = (out = Dirichlet([2.0, 3.0]), a = PointMass([2.0, 1.0])),) =>
                -loggamma(3.0) + loggamma(2.0) + loggamma(1.0) - (2.0 - 1) * (digamma(2.0) - digamma(5.0)),
        ],
    )
end

@testitem "rules:GammaShapeRate:α-β-marginals" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    using SpecialFunctions: loggamma

    # E[log β] under GammaShapeRate(1, 1) is -γₑ, the Euler–Mascheroni constant.
    γₑ = 0.5772156649015315
    @test_message_update_rule(
        node = GammaShapeRate, target = :α,
        cases = [
            (q = (out = GammaShapeRate(1.0, 1.0), β = GammaShapeRate(1.0, 1.0)),) => GammaShapeLikelihood(1.0, -2γₑ),
            (q = (out = PointMass(1.0), β = GammaShapeRate(1.0, 1.0)),) => GammaShapeLikelihood(1.0, -γₑ),
        ],
    )
    @test_message_update_rule(
        node = GammaShapeRate, target = :β,
        cases = [
            (q = (out = GammaShapeRate(1.0, 1.0), α = GammaShapeRate(1.0, 1.0)),) => GammaShapeRate(2.0, 1.0),
            (q = (out = PointMass(1.0), α = GammaShapeRate(1.0, 1.0)),) => GammaShapeRate(2.0, 1.0),
            (q = (out = GammaShapeScale(1.0, 1.0), α = PointMass(10.0)),) => GammaShapeRate(11.0, 1.0),
            (q = (out = GammaShapeScale(1.0, 10.0), α = GammaShapeRate(1.0, 1.0)),) => GammaShapeRate(2.0, 10.0),
        ],
    )
    @test_marginal_update_rule(
        node = GammaShapeRate, target = (:out, :α, :β),
        cases = [
            (m = (out = GammaShapeRate(1.0, 2.0), α = PointMass(1.0), β = PointMass(2.0)),) =>
                FactorizedCluster((:out,) => GammaShapeRate(1.0, 4.0), (:α,) => PointMass(1.0), (:β,) => PointMass(2.0)),
            (m = (out = GammaShapeScale(2.0, 2.0), α = PointMass(2.0), β = PointMass(3.0)),) =>
                FactorizedCluster((:out,) => GammaShapeRate(3.0, 3.5), (:α,) => PointMass(2.0), (:β,) => PointMass(3.0)),
        ],
    )

    # Two shape likelihoods multiply by adding their parameters.
    @test prod(BayesBase.PreserveTypeProd(Distribution), GammaShapeLikelihood(1.0, 2.0), GammaShapeLikelihood(2.0, 0.5)) == GammaShapeLikelihood(3.0, 2.5)
    @test logpdf(GammaShapeLikelihood(1.0, 2.0), 3.0) ≈ 2.0 * 3.0 - loggamma(3.0)
end

@testitem "rules:Categorical:marginals" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    using BayesBase: tiny

    @test_marginal_update_rule(
        node = Categorical, target = (:out, :p), float_types = (Float32, Float64),
        cases = [
            (m = (out = PointMass([0.0, 1.0]), p = Dirichlet([2.0, 1.0])),) => FactorizedCluster((:out,) => PointMass([0.0, 1.0]), (:p,) => Dirichlet([2.0, 2.0])),
            (m = (out = PointMass([1.0, 0.0]), p = Dirichlet([1.0, 2.0])),) => FactorizedCluster((:out,) => PointMass([1.0, 0.0]), (:p,) => Dirichlet([2.0, 2.0])),
        ],
    )
    @test_marginal_update_rule(
        node = Categorical, target = (:out, :p), float_types = (Float32, Float64),
        cases = [
            (m = (out = Categorical([0.2, 0.8]), p = PointMass([0.0, 1.0])),) =>
                FactorizedCluster((:out,) => Categorical([tiny, 0.8] ./ (tiny + 0.8)), (:p,) => PointMass([0.0, 1.0])),
            (m = (out = Categorical([0.8, 0.2]), p = PointMass([1.0, 0.0])),) =>
                FactorizedCluster((:out,) => Categorical([0.8, tiny] ./ (0.8 + tiny)), (:p,) => PointMass([1.0, 0.0])),
        ],
    )

    # Towards `p` from anything but a Categorical or a one-hot point mass is an error.
    @test_throws ArgumentError getresult(call_message_update_rule(Categorical, :p; q = (out = PointMass(1.0),)))
    @test_throws ArgumentError getresult(call_message_update_rule(Categorical, :p; q = (out = 1.0,)))
end

@testitem "rules:Dirichlet:marginals" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions

    @test_marginal_update_rule(
        node = Dirichlet, target = (:out, :a),
        cases = [
            (m = (out = Dirichlet([1.0, 2.0]), a = PointMass([0.2, 1.0])),) => FactorizedCluster((:out,) => Dirichlet([0.2, 2.0]), (:a,) => PointMass([0.2, 1.0])),
            (m = (out = Dirichlet([2.0, 2.0]), a = PointMass([2.0, 0.5])),) => FactorizedCluster((:out,) => Dirichlet([3.0, 1.5]), (:a,) => PointMass([2.0, 0.5])),
            (m = (out = Dirichlet([2.0, 3.0]), a = PointMass([3.0, 1.0])),) => FactorizedCluster((:out,) => Dirichlet([4.0, 3.0]), (:a,) => PointMass([3.0, 1.0])),
        ],
    )
end
