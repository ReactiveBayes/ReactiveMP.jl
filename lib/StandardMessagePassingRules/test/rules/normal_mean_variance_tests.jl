# NormalMeanVariance: where `q_v` is not a point mass, the variance it contributes is
# `1/E[1/v]`, not `E[v]`. For
# `InverseGamma(3, 4)`, `E[1/v] = 3/4`.

@testitem "rules:NormalMeanVariance:out" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions

    @test_message_update_rule(
        node = NormalMeanVariance, target = :out,
        cases = [
            (m = (μ = PointMass(-1.0), v = PointMass(2.0)),) => NormalMeanVariance(-1.0, 2.0),
            (m = (μ = PointMass(2.0), v = PointMass(1.0)),) => NormalMeanVariance(2.0, 1.0),
            (m = (μ = NormalMeanVariance(0.0, 1.0), v = PointMass(2.0)),) => ExpectedWithLogScale(NormalMeanVariance(0.0, 3.0), 0),
            (m = (μ = NormalMeanPrecision(2.0, 0.5), v = PointMass(1.0)),) => ExpectedWithLogScale(NormalMeanVariance(2.0, 3.0), 0),
            (m = (μ = NormalWeightedMeanPrecision(2.0, 0.5), v = PointMass(1.0)),) => ExpectedWithLogScale(NormalMeanVariance(4.0, 3.0), 0),
            (q = (μ = PointMass(1.0), v = PointMass(2.0)),) => NormalMeanVariance(1.0, 2.0),
            (q = (μ = NormalMeanVariance(1.0, 2.0), v = InverseGamma(3.0, 4.0)),) => NormalMeanVariance(1.0, 4 / 3),
            (m = (μ = PointMass(-1.0),), q = (v = InverseGamma(3.0, 4.0),)) => NormalMeanVariance(-1.0, 4 / 3),
            (m = (μ = NormalMeanVariance(0.0, 1.0),), q = (v = InverseGamma(3.0, 4.0),)) => NormalMeanVariance(0.0, 1 + 4 / 3),
            (m = (μ = NormalMeanVariance(2.0, 0.5),), q = (v = PointMass(1.0),)) => ExpectedWithLogScale(NormalMeanVariance(2.0, 1.5), 0),
        ],
    )
end

@testitem "rules:NormalMeanVariance:μ" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions

    @test_message_update_rule(
        node = NormalMeanVariance, target = :μ,
        cases = [
            (m = (out = PointMass(-1.0), v = PointMass(2.0)),) => NormalMeanVariance(-1.0, 2.0),
            (m = (out = NormalMeanVariance(0.0, 1.0), v = PointMass(2.0)),) => ExpectedWithLogScale(NormalMeanVariance(0.0, 3.0), 0),
            (m = (out = NormalWeightedMeanPrecision(2.0, 0.5), v = PointMass(1.0)),) => ExpectedWithLogScale(NormalMeanVariance(4.0, 3.0), 0),
            (q = (out = PointMass(1.0), v = PointMass(2.0)),) => NormalMeanVariance(1.0, 2.0),
            (q = (out = NormalMeanVariance(1.0, 2.0), v = InverseGamma(3.0, 4.0)),) => NormalMeanVariance(1.0, 4 / 3),
            (m = (out = PointMass(-1.0),), q = (v = InverseGamma(3.0, 4.0),)) => NormalMeanVariance(-1.0, 4 / 3),
            # No log scale: see `rules/normal_mean_variance/mean.jl`.
            (m = (out = NormalMeanVariance(0.0, 1.0),), q = (v = InverseGamma(3.0, 4.0),)) => NormalMeanVariance(0.0, 1 + 4 / 3),
        ],
    )
end

@testitem "rules:NormalMeanVariance:marginals" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions

    # xi = (1, 2/0.5), W_out = 1, W_μ = 2, and E[1/v] = 1/2 couples them.
    @test_marginal_update_rule(
        node = NormalMeanVariance, target = (:out, :μ),
        cases = [
            (m = (out = NormalMeanVariance(1.0, 1.0), μ = NormalMeanVariance(2.0, 0.5)), q = (v = PointMass(2.0),)) =>
                MvNormalWeightedMeanPrecision([1.0, 4.0], [1.5 -0.5; -0.5 2.5]),
        ],
    )
end

@testitem "rules:NormalMeanVariance:average-energy" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions
    using StatsFuns: log2π

    @test_average_energy(
        node = NormalMeanVariance,
        cases = [
            (q = (out = NormalMeanVariance(0.0, 1.0), μ = NormalMeanVariance(1.0, 2.0), v = PointMass(2.0)),) =>
                (log2π + log(2.0) + (2.0 + 1.0 + 1.0) / 2) / 2,
            (clusters = ((:out, :μ) => MvNormalMeanCovariance([0.0, 1.0], [1.0 0.2; 0.2 2.0]),), q = (v = PointMass(2.0),)) =>
                (log2π + log(2.0) + (1.0 + 2.0 - 0.4 + 1.0) / 2) / 2,
        ],
    )
end

@testitem "rules:NormalMeanVariance:verification" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions

    # Against the node's own log-density, not against stored numbers: belief propagation and
    # variational inputs, the latter checking the variance 1/E[1/v]. Rules mixing
    # messages and marginals are checked by their tables only: the verification tool takes
    # one kind of input or the other.
    @verify_message_update_rule(node = NormalMeanVariance, target = :out, m = (μ = NormalMeanVariance(0.5, 1.5), v = PointMass(2.0)))
    @verify_message_update_rule(node = NormalMeanVariance, target = :μ, m = (out = NormalMeanVariance(-1.0, 0.5), v = PointMass(3.0)))
    @verify_message_update_rule(node = NormalMeanVariance, target = :out, q = (μ = NormalMeanVariance(1.0, 2.0), v = InverseGamma(3.0, 4.0)))
    @verify_message_update_rule(node = NormalMeanVariance, target = :μ, q = (out = NormalMeanVariance(1.0, 2.0), v = InverseGamma(3.0, 4.0)))
end

@testitem "rules:NormalMeanVariance:v" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions

    # Belief propagation gives a log-density on the half line, compared by evaluating it:
    # -log(s + v)/2 - (a - b)²/(2(s + v)) with s the variance the normal messages carry.
    likelihood(a, b, s) = (v) -> -log(s + v) / 2 - (a - b)^2 / (2 * (s + v))
    cases = [
        (m = (out = PointMass(2.0), μ = NormalMeanVariance(0.0, 1.0)), expected = likelihood(2.0, 0.0, 1.0)),
        (m = (out = NormalMeanVariance(0.5, 2.0), μ = PointMass(-3.5)), expected = likelihood(0.5, -3.5, 2.0)),
        (m = (out = NormalMeanVariance(1.0, 0.5), μ = NormalMeanVariance(-1.0, 2.0)), expected = likelihood(1.0, -1.0, 2.5)),
    ]
    for case in cases
        message = getresult(call_message_update_rule(NormalMeanVariance, :v; m = case.m))
        @test message isa ContinuousUnivariateLogPdf
        @test all(v -> logpdf(message, v) ≈ case.expected(v), (0.1, 1.0, 3.5, 10.0))
    end

    # Variational: an unchecked inverse gamma of shape -1/2.
    @test_message_update_rule(
        node = NormalMeanVariance, target = :v, check_type_promotion = false,
        cases = [
            (q = (out = PointMass(-1.0), μ = PointMass(2.0)),) => GammaInverse(-0.5, 4.5; check_args = false),
            (q = (out = NormalMeanVariance(-1.0, 2.0), μ = PointMass(2.0)),) => GammaInverse(-0.5, 5.5; check_args = false),
            (q = (out = PointMass(3.0), μ = NormalMeanPrecision(1.0, 4.0)),) => GammaInverse(-0.5, 2.125; check_args = false),
        ],
    )
    @test_message_update_rule(
        node = NormalMeanVariance, target = :v, check_type_promotion = false,
        cases = [
            (clusters = ((:out, :μ) => MvNormalMeanCovariance([1.0, 1.0], [1.0 0.0; 0.0 1.0]),),) => GammaInverse(-0.5, 1.0; check_args = false),
            (clusters = ((:out, :μ) => MvNormalMeanCovariance([2.0, 3.0], [2.0 -0.1; -0.1 3.0]),),) => GammaInverse(-0.5, 3.1; check_args = false),
            (clusters = ((:out, :μ) => MvNormalMeanCovariance([4.0, 1.0], [4.0 1.0; 1.0 9.0]),),) => GammaInverse(-0.5, 10.0; check_args = false),
        ],
    )
end

@testitem "rules:NormalMeanVariance:split-marginals" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions

    # A point-mass message splits the cluster. Towards the other member: the Normal the
    # point mass implies, times its own message. With a point-mass q_v, 1/E[1/v] = E[v].
    @test_marginal_update_rule(
        node = NormalMeanVariance, target = (:out, :μ),
        cases = [
            (m = (out = PointMass(1.0), μ = NormalMeanVariance(0.0, 1.0)), q = (v = PointMass(2.0),)) =>
                FactorizedCluster((:out,) => PointMass(1.0), (:μ,) => NormalWeightedMeanPrecision(0.5, 1.5)),
            (m = (out = NormalMeanVariance(0.0, 1.0), μ = PointMass(1.0)), q = (v = PointMass(2.0),)) =>
                FactorizedCluster((:out,) => NormalWeightedMeanPrecision(0.5, 1.5), (:μ,) => PointMass(1.0)),
            # An inverse gamma q_v, InverseGamma(3, 4): E[1/v] = 3/4, so the precision added is 3/4.
            (m = (out = PointMass(1.0), μ = NormalMeanVariance(0.0, 1.0)), q = (v = InverseGamma(3.0, 4.0),)) =>
                FactorizedCluster((:out,) => PointMass(1.0), (:μ,) => NormalWeightedMeanPrecision(0.75, 1.75)),
        ],
    )
    @test_marginal_update_rule(
        node = NormalMeanVariance, target = (:out, :μ, :v),
        cases = [
            (m = (out = NormalMeanVariance(0.0, 1.0), μ = PointMass(1.0), v = PointMass(2.0)),) =>
                FactorizedCluster((:out,) => NormalWeightedMeanPrecision(0.5, 1.5), (:μ,) => PointMass(1.0), (:v,) => PointMass(2.0)),
            (m = (out = PointMass(1.0), μ = NormalMeanVariance(0.0, 1.0), v = PointMass(2.0)),) =>
                FactorizedCluster((:out,) => PointMass(1.0), (:μ,) => NormalWeightedMeanPrecision(0.5, 1.5), (:v,) => PointMass(2.0)),
            (m = (out = NormalMeanVariance(1.0, 1.0), μ = NormalMeanVariance(2.0, 0.5), v = PointMass(2.0)),) =>
                FactorizedCluster((:out, :μ) => MvNormalWeightedMeanPrecision([1.0, 4.0], [1.5 -0.5; -0.5 2.5]), (:v,) => PointMass(2.0)),
        ],
    )
end
