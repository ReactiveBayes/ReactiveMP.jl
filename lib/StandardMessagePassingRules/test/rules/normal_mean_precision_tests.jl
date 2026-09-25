# NormalMeanPrecision's variational rules take E[τ], which is right for a precision: naive VMP,
# exp E_q[log N(out | μ, 1/τ)], is linear in τ. So unlike NormalMeanVariance there is nothing
# to correct. `GammaShapeRate(3, 2)` has mean 3/2.

@testitem "rules:NormalMeanPrecision:out" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions

    @test_message_update_rule(
        node = NormalMeanPrecision, target = :out,
        cases = [
            (m = (μ = PointMass(1.0), τ = PointMass(2.0)),) => NormalMeanPrecision(1.0, 2.0),
            (m = (μ = NormalMeanVariance(0.0, 1.0), τ = PointMass(2.0)),) => ExpectedWithLogScale(NormalMeanPrecision(0.0, 2 / 3), 0),
            (q = (μ = PointMass(1.0), τ = PointMass(2.0)),) => NormalMeanPrecision(1.0, 2.0),
            (q = (μ = NormalMeanVariance(1.0, 2.0), τ = GammaShapeRate(3.0, 2.0)),) => NormalMeanPrecision(1.0, 1.5),
            (m = (μ = PointMass(-1.0),), q = (τ = GammaShapeRate(3.0, 2.0),)) => NormalMeanPrecision(-1.0, 1.5),
            (m = (μ = NormalMeanVariance(0.0, 1.0),), q = (τ = GammaShapeRate(3.0, 2.0),)) => NormalMeanPrecision(0.0, 0.6),
        ],
    )
end

@testitem "rules:NormalMeanPrecision:μ" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions

    @test_message_update_rule(
        node = NormalMeanPrecision, target = :μ,
        cases = [
            (m = (out = PointMass(1.0), τ = PointMass(2.0)),) => NormalMeanPrecision(1.0, 2.0),
            (m = (out = NormalMeanVariance(0.0, 1.0), τ = PointMass(2.0)),) => ExpectedWithLogScale(NormalMeanVariance(0.0, 1.5), 0),
            (q = (out = PointMass(1.0), τ = PointMass(2.0)),) => NormalMeanPrecision(1.0, 2.0),
            (q = (out = NormalMeanVariance(1.0, 2.0), τ = GammaShapeRate(3.0, 2.0)),) => NormalMeanPrecision(1.0, 1.5),
            (m = (out = PointMass(-1.0),), q = (τ = GammaShapeRate(3.0, 2.0),)) => NormalMeanPrecision(-1.0, 1.5),
            (m = (out = NormalMeanVariance(0.0, 1.0),), q = (τ = GammaShapeRate(3.0, 2.0),)) => NormalMeanVariance(0.0, 1 + 2 / 3),
        ],
    )
end

@testitem "rules:NormalMeanPrecision:τ" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions

    @test_message_update_rule(
        node = NormalMeanPrecision, target = :τ,
        cases = [
            # θ = 2 / (1 + 2 + 1²)
            (q = (out = NormalMeanVariance(1.0, 1.0), μ = NormalMeanVariance(0.0, 2.0)),) => Gamma(1.5, 0.5),
            # θ = 2 / (1 - 2·0.5 + 2 + 1²)
            (clusters = ((:out, :μ) => MvNormalMeanCovariance([1.0, 0.0], [1.0 0.5; 0.5 2.0]),),) => Gamma(1.5, 2 / 3),
        ],
    )
end

@testitem "rules:NormalMeanPrecision:marginals" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions

    @test_marginal_update_rule(
        node = NormalMeanPrecision, target = (:out, :μ),
        cases = [
            (m = (out = NormalMeanVariance(1.0, 1.0), μ = NormalMeanVariance(2.0, 0.5)), q = (τ = PointMass(2.0),)) =>
                MvNormalWeightedMeanPrecision([1.0, 4.0], [3.0 -2.0; -2.0 4.0]),
        ],
    )
end

@testitem "rules:NormalMeanPrecision:average-energy" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions
    using StatsFuns: log2π

    @test_average_energy(
        node = NormalMeanPrecision,
        cases = [
            (q = (out = NormalMeanVariance(0.0, 1.0), μ = NormalMeanVariance(1.0, 2.0), τ = PointMass(2.0)),) =>
                (log2π - log(2.0) + 2.0 * (2.0 + 1.0 + 1.0)) / 2,
            (clusters = ((:out, :μ) => MvNormalMeanCovariance([0.0, 1.0], [1.0 0.2; 0.2 2.0]),), q = (τ = PointMass(2.0),)) =>
                (log2π - log(2.0) + 2.0 * (1.0 + 2.0 - 0.4 + 1.0)) / 2,
        ],
    )
end

@testitem "rules:NormalMeanPrecision:verification" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions

    @verify_message_update_rule(node = NormalMeanPrecision, target = :out, m = (μ = NormalMeanVariance(0.5, 1.5), τ = PointMass(2.0)))
    @verify_message_update_rule(node = NormalMeanPrecision, target = :μ, m = (out = NormalMeanVariance(-1.0, 0.5), τ = PointMass(3.0)))
    @verify_message_update_rule(node = NormalMeanPrecision, target = :out, q = (μ = NormalMeanVariance(1.0, 2.0), τ = GammaShapeRate(3.0, 2.0)))
    @verify_message_update_rule(node = NormalMeanPrecision, target = :μ, q = (out = NormalMeanVariance(1.0, 2.0), τ = GammaShapeRate(3.0, 2.0)))
    @verify_message_update_rule(node = NormalMeanPrecision, target = :τ, q = (out = NormalMeanVariance(1.0, 1.0), μ = NormalMeanVariance(0.0, 2.0)))
end

@testitem "rules:NormalMeanPrecision:split-marginals" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions

    # A point-mass message splits the cluster; the precision adds E[τ].
    @test_marginal_update_rule(
        node = NormalMeanPrecision, target = (:out, :μ),
        cases = [
            (m = (out = PointMass(1.0), μ = NormalMeanPrecision(0.0, 1.0)), q = (τ = GammaShapeRate(3.0, 2.0),)) =>
                FactorizedCluster((:out,) => PointMass(1.0), (:μ,) => NormalWeightedMeanPrecision(1.5, 2.5)),
            (m = (out = NormalMeanPrecision(0.0, 1.0), μ = PointMass(1.0)), q = (τ = PointMass(2.0),)) =>
                FactorizedCluster((:out,) => NormalWeightedMeanPrecision(2.0, 3.0), (:μ,) => PointMass(1.0)),
        ],
    )
    @test_marginal_update_rule(
        node = NormalMeanPrecision, target = (:out, :μ, :τ),
        cases = [
            (m = (out = NormalMeanPrecision(0.0, 1.0), μ = PointMass(1.0), τ = PointMass(2.0)),) =>
                FactorizedCluster((:out,) => NormalWeightedMeanPrecision(2.0, 3.0), (:μ,) => PointMass(1.0), (:τ,) => PointMass(2.0)),
            (m = (out = PointMass(1.0), μ = NormalMeanPrecision(0.0, 1.0), τ = PointMass(2.0)),) =>
                FactorizedCluster((:out,) => PointMass(1.0), (:μ,) => NormalWeightedMeanPrecision(2.0, 3.0), (:τ,) => PointMass(2.0)),
            (m = (out = NormalMeanVariance(1.0, 1.0), μ = NormalMeanVariance(2.0, 0.5), τ = PointMass(0.5)),) =>
                FactorizedCluster((:out, :μ) => MvNormalWeightedMeanPrecision([1.0, 4.0], [1.5 -0.5; -0.5 2.5]), (:τ,) => PointMass(0.5)),
        ],
    )
end
