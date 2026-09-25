# MvNormalMeanCovariance: tables of cases, and hand-derived ones for the marginal rules and for
# a non-point-mass q_Σ, where naive VMP gives E[Σ⁻¹]⁻¹. For Σ ~ InverseWishart(ν, Ψ),
# E[Σ⁻¹] = νΨ⁻¹.

@testitem "rules:MvNormalMeanCovariance:out-μ" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions

    for target in (:out, :μ)
        other = target === :out ? :μ : :out
        m(value, Σ) = NamedTuple{(other, :Σ)}((value, Σ))
        @test_message_update_rule(
            node = MvNormalMeanCovariance, target = target,
            cases = [
                (m = m(PointMass([1.0, 3.0]), PointMass([3.0 2.0; 2.0 4.0])),) => MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0]),
                (m = m(MvNormalMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), PointMass([6.0 4.0; 4.0 8.0])),) =>
                    ExpectedWithAnnotations(MvNormalMeanCovariance([2.0, 1.0], [13 / 2 15 / 4; 15 / 4 67 / 8]); logscale = 0),
                (m = m(MvNormalMeanCovariance([0.0, 0.0], [7.0 -1.0; -1.0 9.0]), PointMass([12.0 -2.0; -2.0 7.0])),) =>
                    ExpectedWithAnnotations(MvNormalMeanCovariance([0.0, 0.0], [19.0 -3.0; -3.0 16.0]); logscale = 0),
                (q = m(PointMass([-1.0, 2.0]), PointMass([7.0 -1.0; -1.0 9.0])),) => MvNormalMeanCovariance([-1.0, 2.0], [7.0 -1.0; -1.0 9.0]),
                (q = m(MvNormalMeanCovariance([1.0, 2.0], [3.0 2.0; 2.0 4.0]), PointMass([2.0 0.0; 0.0 2.0])),) => MvNormalMeanCovariance([1.0, 2.0], [2.0 0.0; 0.0 2.0]),
                # E[Σ⁻¹] = 5 · (2I)⁻¹, so the covariance is 0.4 I, not E[Σ] = I.
                (q = m(MvNormalMeanCovariance([1.0, 2.0], [3.0 2.0; 2.0 4.0]), InverseWishart(5.0, [2.0 0.0; 0.0 2.0])),) => MvNormalMeanCovariance([1.0, 2.0], [0.4 0.0; 0.0 0.4]),
            ],
        )
        mixed(value) = (m = NamedTuple{(other,)}((value,)), q = (Σ = InverseWishart(5.0, [2.0 0.0; 0.0 2.0]),))
        @test_message_update_rule(
            node = MvNormalMeanCovariance, target = target,
            cases = [
                mixed(PointMass([1.0, 3.0])) => MvNormalMeanCovariance([1.0, 3.0], [0.4 0.0; 0.0 0.4]),
                mixed(MvNormalMeanCovariance([1.0, 3.0], [1.0 0.0; 0.0 2.0])) => MvNormalMeanCovariance([1.0, 3.0], [1.4 0.0; 0.0 2.4]),
            ],
        )
    end
end

@testitem "rules:MvNormalMeanCovariance:Σ" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions
    import ExponentialFamily: InverseWishartFast

    @test_message_update_rule(
        node = MvNormalMeanCovariance, target = :Σ,
        cases = [
            (q = (out = PointMass([1.0, 2.0]), μ = MvNormalMeanCovariance([3.0, 5.0], [3.0 2.0; 2.0 4.0])),) => InverseWishartFast(-2.0, [7.0 8.0; 8.0 13.0]),
            (q = (out = MvNormalMeanCovariance([1.0, 2.0], [3.0 2.0; 2.0 4.0]), μ = MvNormalMeanCovariance([3.0, 5.0], [3.0 2.0; 2.0 4.0])),) => InverseWishartFast(-2.0, [10.0 10.0; 10.0 17.0]),
            (q = (out = MvNormalMeanPrecision([1.0, 2.0], [3.0 2.0; 2.0 4.0]), μ = PointMass([3.0, 5.0])),) => InverseWishartFast(-2.0, [9 / 2 23 / 4; 23 / 4 75 / 8]),
            (clusters = ((:out, :μ) => MvNormalMeanCovariance(ones(4), [1.0 0 0 0; 0 1.0 0 0; 0 0 1.0 0; 0 0 0 1.0]),),) => InverseWishartFast(-2.0, [2.0 0.0; 0.0 2.0]),
        ],
    )
end

@testitem "rules:MvNormalMeanCovariance:marginals" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions

    I2 = [1.0 0.0; 0.0 1.0]
    # Two messages with precision I and a point-mass Σ = 2I, so W̄ = I/2.
    joint = MvNormalWeightedMeanPrecision([1.0, 2.0, 3.0, 4.0], [1.5 0 -0.5 0; 0 1.5 0 -0.5; -0.5 0 1.5 0; 0 -0.5 0 1.5])
    @test_marginal_update_rule(
        node = MvNormalMeanCovariance, target = (:out, :μ),
        cases = [
            (m = (out = MvNormalWeightedMeanPrecision([1.0, 2.0], I2), μ = MvNormalWeightedMeanPrecision([3.0, 4.0], I2)), q = (Σ = PointMass(2 * I2),)) => joint,
            # A point mass on out: μ's block is N(out, Σ) times its message, precision I + I/2.
            (m = (out = PointMass([1.0, 1.0]), μ = MvNormalWeightedMeanPrecision([3.0, 4.0], I2)), q = (Σ = PointMass(2 * I2),)) =>
                FactorizedCluster((:out,) => PointMass([1.0, 1.0]), (:μ,) => MvNormalWeightedMeanPrecision([3.5, 4.5], 1.5 * I2)),
            # An inverse Wishart q_Σ: its E[Σ⁻¹] = 5/2 I is the precision it adds.
            (m = (out = MvNormalWeightedMeanPrecision([3.0, 4.0], I2), μ = PointMass([1.0, 1.0])), q = (Σ = InverseWishart(5.0, 2 * I2),)) =>
                FactorizedCluster((:out,) => MvNormalWeightedMeanPrecision([5.5, 6.5], 3.5 * I2), (:μ,) => PointMass([1.0, 1.0])),
        ],
    )
    @test_marginal_update_rule(
        node = MvNormalMeanCovariance, target = (:out, :μ, :Σ),
        cases = [
            (m = (out = MvNormalWeightedMeanPrecision([1.0, 2.0], I2), μ = MvNormalWeightedMeanPrecision([3.0, 4.0], I2), Σ = PointMass(2 * I2)),) =>
                FactorizedCluster((:out, :μ) => joint, (:Σ,) => PointMass(2 * I2)),
            (m = (out = MvNormalWeightedMeanPrecision([1.0, 2.0], I2), μ = PointMass([1.0, 1.0]), Σ = PointMass(2 * I2)),) =>
                FactorizedCluster((:out,) => MvNormalWeightedMeanPrecision([1.5, 2.5], 1.5 * I2), (:μ,) => PointMass([1.0, 1.0]), (:Σ,) => PointMass(2 * I2)),
            (m = (out = PointMass([1.0, 1.0]), μ = MvNormalWeightedMeanPrecision([3.0, 4.0], I2), Σ = PointMass(2 * I2)),) =>
                FactorizedCluster((:out,) => PointMass([1.0, 1.0]), (:μ,) => MvNormalWeightedMeanPrecision([3.5, 4.5], 1.5 * I2), (:Σ,) => PointMass(2 * I2)),
        ],
    )
end

@testitem "rules:MvNormalMeanCovariance:average-energy" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    using LinearAlgebra: logdet
    using StatsFuns: log2π

    I2 = [1.0 0.0; 0.0 1.0]
    # (2 log 2π + log 4 + tr((2I)⁻¹ · I)) / 2, in each representation of q_μ.
    for q_μ in (MvNormalMeanCovariance([1.0, 1.0], I2), MvNormalMeanPrecision([1.0, 1.0], I2), MvNormalWeightedMeanPrecision([1.0, 1.0], I2))
        @test call_average_energy(MvNormalMeanCovariance; q = (out = PointMass([1.0, 1.0]), μ = q_μ, Σ = PointMass(2 * I2))) ≈ 3.0310242469692907
    end
    # The joint's form: the same difference moment from the blocks of q(out, μ).
    joint = MvNormalMeanCovariance([1.0, 1.0, 1.0, 1.0], [1.0 0 0 0; 0 1.0 0 0; 0 0 1.0 0; 0 0 0 1.0])
    @test call_average_energy(MvNormalMeanCovariance; q = (Σ = PointMass(2 * I2),), clusters = ((:out, :μ) => joint,)) ≈ (2 * log2π + log(4.0) + 2.0) / 2
    # An inverse Wishart q_Σ: E[log |Σ|] and E[Σ⁻¹] = νΨ⁻¹ both enter.
    q_Σ = InverseWishart(5.0, 2 * I2)
    @test call_average_energy(MvNormalMeanCovariance; q = (out = PointMass([1.0, 1.0]), μ = MvNormalMeanCovariance([1.0, 1.0], I2), Σ = q_Σ)) ≈
        (2 * log2π + mean(logdet, q_Σ) + (5 / 2) * 2) / 2
end
