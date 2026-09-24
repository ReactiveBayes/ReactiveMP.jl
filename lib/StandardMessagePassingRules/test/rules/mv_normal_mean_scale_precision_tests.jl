# MvNormalMeanScalePrecision, the multivariate normal with precision γI: v6's tables, and
# hand-derived cases for the marginal rules and the energy.

@testitem "rules:MvNormalMeanScalePrecision:out-μ-γ" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions

    @test_message_update_rule(
        node = MvNormalMeanScalePrecision, target = :out,
        cases = [
            (q = (μ = MvNormalMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), γ = Gamma(3.0, 2.0)),) => MvNormalMeanScalePrecision([2.0, 1.0], 6.0),
            (q = (μ = MvNormalWeightedMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), γ = Gamma(4.0, 2.0)),) => MvNormalMeanScalePrecision([3 / 4, -1 / 8], 8.0),
            (m = (μ = MvNormalMeanCovariance([0.0, 1.0], [2.0 -1.0; -1.0 4.0]),), q = (γ = Gamma(3.0, 1.0),)) => MvNormalMeanCovariance([0.0, 1.0], [7 / 3 -1.0; -1.0 13 / 3]),
            (m = (μ = MvNormalMeanScalePrecision([2.0, 1.0], 3.0),), q = (γ = Gamma(1.0, 1.0),)) => MvNormalMeanScalePrecision([2.0, 1.0], 3.0 * 1.0 / (3.0 + 1.0)),
        ],
    )
    @test_message_update_rule(
        node = MvNormalMeanScalePrecision, target = :μ,
        cases = [
            (q = (out = MvNormalMeanCovariance([2.0, 1.0], [3.0 2.0; 2.0 4.0]), γ = GammaShapeRate(1.0, 1.0)),) => MvNormalMeanScalePrecision([2.0, 1.0], 1.0),
            (m = (out = MvNormalMeanCovariance([0.0, 0.0], [7.0 -1.0; -1.0 9.0]),), q = (γ = GammaShapeRate(4.0, 2.0),)) => MvNormalMeanCovariance([0.0, 0.0], [7.5 -1.0; -1.0 9.5]),
            (m = (out = MvNormalMeanScalePrecision([2.0, 1.0], 3.0),), q = (γ = Gamma(1.0, 1.0),)) => MvNormalMeanScalePrecision([2.0, 1.0], 3.0 * 1.0 / (3.0 + 1.0)),
        ],
    )
    @test_message_update_rule(
        node = MvNormalMeanScalePrecision, target = :γ,
        cases = [
            (q = (out = MvNormalMeanCovariance([1.0, 2.0], [3.0 2.0; 2.0 4.0]), μ = MvNormalMeanCovariance([3.0, 5.0], [3.0 2.0; 2.0 4.0])),) => GammaShapeRate(2.0, 13.5),
            (clusters = ((:out, :μ) => MvNormalMeanCovariance(ones(4), [1.0 0 0 0; 0 1.0 0 0; 0 0 1.0 0; 0 0 0 1.0]),),) => GammaShapeRate(2.0, 2.0),
        ],
    )
end

@testitem "rules:MvNormalMeanScalePrecision:marginals-energy" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    using StatsFuns: log2π
    using SpecialFunctions: digamma

    I2 = [1.0 0.0; 0.0 1.0]
    # Messages with precision I, and E[γ] = 1/2 coupling them.
    joint = MvNormalWeightedMeanPrecision([1.0, 2.0, 3.0, 4.0], [1.5 0 -0.5 0; 0 1.5 0 -0.5; -0.5 0 1.5 0; 0 -0.5 0 1.5])
    @test_marginal_update_rule(
        node = MvNormalMeanScalePrecision, target = (:out, :μ),
        cases = [
            (m = (out = MvNormalWeightedMeanPrecision([1.0, 2.0], I2), μ = MvNormalWeightedMeanPrecision([3.0, 4.0], I2)), q = (γ = PointMass(0.5),)) => joint,
            (m = (out = PointMass([1.0, 1.0]), μ = MvNormalWeightedMeanPrecision([3.0, 4.0], I2)), q = (γ = GammaShapeRate(1.0, 2.0),)) =>
                FactorizedCluster((:out,) => PointMass([1.0, 1.0]), (:μ,) => MvNormalWeightedMeanPrecision([3.5, 4.5], 1.5 * I2)),
            (m = (out = MvNormalWeightedMeanPrecision([3.0, 4.0], I2), μ = PointMass([1.0, 1.0])), q = (γ = PointMass(0.5),)) =>
                FactorizedCluster((:out,) => MvNormalWeightedMeanPrecision([3.5, 4.5], 1.5 * I2), (:μ,) => PointMass([1.0, 1.0])),
        ],
    )
    @test_marginal_update_rule(
        node = MvNormalMeanScalePrecision, target = (:out, :μ, :γ),
        cases = [
            (m = (out = MvNormalWeightedMeanPrecision([1.0, 2.0], I2), μ = MvNormalWeightedMeanPrecision([3.0, 4.0], I2), γ = PointMass(0.5)),) =>
                FactorizedCluster((:out, :μ) => joint, (:γ,) => PointMass(0.5)),
            (m = (out = MvNormalWeightedMeanPrecision([1.0, 2.0], I2), μ = PointMass([1.0, 1.0]), γ = PointMass(0.5)),) =>
                FactorizedCluster((:out,) => MvNormalWeightedMeanPrecision([1.5, 2.5], 1.5 * I2), (:μ,) => PointMass([1.0, 1.0]), (:γ,) => PointMass(0.5)),
            (m = (out = PointMass([1.0, 1.0]), μ = MvNormalWeightedMeanPrecision([3.0, 4.0], I2), γ = PointMass(0.5)),) =>
                FactorizedCluster((:out,) => PointMass([1.0, 1.0]), (:μ,) => MvNormalWeightedMeanPrecision([3.5, 4.5], 1.5 * I2), (:γ,) => PointMass(0.5)),
        ],
    )
    # (d log 2π - d E[log γ] + E[γ] tr S) / 2, with γ ~ GammaShapeRate(2, 4): E[γ] = 1/2 and
    # E[log γ] = digamma(2) - log(4); S = I + I + ΔΔᵀ for Δ = (0.5, -0.5).
    q = (out = MvNormalMeanCovariance([1.5, 0.5], I2), μ = MvNormalMeanCovariance([1.0, 1.0], I2), γ = GammaShapeRate(2.0, 4.0))
    @test call_average_energy(MvNormalMeanScalePrecision; q) ≈ (2 * log2π - 2 * (digamma(2.0) - log(4.0)) + 0.5 * (4.0 + 0.5)) / 2
    whole = MvNormalMeanCovariance([1.0, 1.0, 1.0, 1.0], [1.0 0 0 0; 0 1.0 0 0; 0 0 1.0 0; 0 0 0 1.0])
    @test call_average_energy(MvNormalMeanScalePrecision; q = (γ = PointMass(0.5),), clusters = ((:out, :μ) => whole,)) ≈ (2 * log2π - 2 * log(0.5) + 0.5 * 4.0) / 2
end
