# Cases from v6's `test/rules/normal_mean_variance/`, except where `q_v` is not a point mass:
# there v6 used `E[v]` (ReactiveMP.jl#669) and these expect `1/E[1/v]`. For
# `InverseGamma(3, 4)`, `E[1/v] = 3/4`.

@testitem "rules:NormalMeanVariance:out" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions

    @test_message_update_rule(
        node = NormalMeanVariance, target = :out,
        cases = [
            (m = (μ = PointMass(-1.0), v = PointMass(2.0)),) => NormalMeanVariance(-1.0, 2.0),
            (m = (μ = PointMass(2.0), v = PointMass(1.0)),) => NormalMeanVariance(2.0, 1.0),
            (m = (μ = NormalMeanVariance(0.0, 1.0), v = PointMass(2.0)),) => ExpectedWithAnnotations(NormalMeanVariance(0.0, 3.0); logscale = 0),
            (m = (μ = NormalMeanPrecision(2.0, 0.5), v = PointMass(1.0)),) => ExpectedWithAnnotations(NormalMeanVariance(2.0, 3.0); logscale = 0),
            (m = (μ = NormalWeightedMeanPrecision(2.0, 0.5), v = PointMass(1.0)),) => ExpectedWithAnnotations(NormalMeanVariance(4.0, 3.0); logscale = 0),
            (q = (μ = PointMass(1.0), v = PointMass(2.0)),) => NormalMeanVariance(1.0, 2.0),
            (q = (μ = NormalMeanVariance(1.0, 2.0), v = InverseGamma(3.0, 4.0)),) => NormalMeanVariance(1.0, 4 / 3),
            (m = (μ = PointMass(-1.0),), q = (v = InverseGamma(3.0, 4.0),)) => NormalMeanVariance(-1.0, 4 / 3),
            (m = (μ = NormalMeanVariance(0.0, 1.0),), q = (v = InverseGamma(3.0, 4.0),)) => NormalMeanVariance(0.0, 1 + 4 / 3),
            (m = (μ = NormalMeanVariance(2.0, 0.5),), q = (v = PointMass(1.0),)) => ExpectedWithAnnotations(NormalMeanVariance(2.0, 1.5); logscale = 0),
        ],
    )
end

@testitem "rules:NormalMeanVariance:μ" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions

    @test_message_update_rule(
        node = NormalMeanVariance, target = :μ,
        cases = [
            (m = (out = PointMass(-1.0), v = PointMass(2.0)),) => NormalMeanVariance(-1.0, 2.0),
            (m = (out = NormalMeanVariance(0.0, 1.0), v = PointMass(2.0)),) => ExpectedWithAnnotations(NormalMeanVariance(0.0, 3.0); logscale = 0),
            (m = (out = NormalWeightedMeanPrecision(2.0, 0.5), v = PointMass(1.0)),) => ExpectedWithAnnotations(NormalMeanVariance(4.0, 3.0); logscale = 0),
            (q = (out = PointMass(1.0), v = PointMass(2.0)),) => NormalMeanVariance(1.0, 2.0),
            (q = (out = NormalMeanVariance(1.0, 2.0), v = InverseGamma(3.0, 4.0)),) => NormalMeanVariance(1.0, 4 / 3),
            (m = (out = PointMass(-1.0),), q = (v = InverseGamma(3.0, 4.0),)) => NormalMeanVariance(-1.0, 4 / 3),
            # No log scale, as in v6: see `rules/normal_mean_variance/mean.jl`.
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
    # variational inputs. The variational cases are the ones v6 got wrong. Rules mixing
    # messages and marginals are checked by their tables only: the verification tool takes
    # one kind of input or the other.
    @verify_message_update_rule(node = NormalMeanVariance, target = :out, m = (μ = NormalMeanVariance(0.5, 1.5), v = PointMass(2.0)))
    @verify_message_update_rule(node = NormalMeanVariance, target = :μ, m = (out = NormalMeanVariance(-1.0, 0.5), v = PointMass(3.0)))
    @verify_message_update_rule(node = NormalMeanVariance, target = :out, q = (μ = NormalMeanVariance(1.0, 2.0), v = InverseGamma(3.0, 4.0)))
    @verify_message_update_rule(node = NormalMeanVariance, target = :μ, q = (out = NormalMeanVariance(1.0, 2.0), v = InverseGamma(3.0, 4.0)))
end
