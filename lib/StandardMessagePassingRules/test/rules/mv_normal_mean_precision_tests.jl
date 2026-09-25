# MvNormalMeanPrecision: tables of cases, and hand-derived ones for the marginal rules. A q_Λ
# contributes the precision E[Λ]; for Wishart(ν, S), E[Λ] = νS.

@testitem "rules:MvNormalMeanPrecision:out-μ" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions

    for target in (:out, :μ)
        other = target === :out ? :μ : :out
        m(value, Λ) = NamedTuple{(other, :Λ)}((value, Λ))
        # The Wishart cases are checked in Float64 only, not in BigFloat and Float32, which go
        # through a Wishart's Cholesky factor.
        @test_message_update_rule(
            node = MvNormalMeanPrecision, target = target,
            cases = [
                (m = m(PointMass([1.0, 3.0]), PointMass([3.0 2.0; 2.0 4.0])),) => MvNormalMeanPrecision([1.0, 3.0], [3.0 2.0; 2.0 4.0]),
                (m = m(MvNormalMeanCovariance([2.0, 1.0], [3.0 2.0; 2.0 4.0]), PointMass([6.0 4.0; 4.0 8.0] ./ 4)),) =>
                    ExpectedWithAnnotations(MvNormalMeanCovariance([2.0, 1.0], [3.0 2.0; 2.0 4.0] + inv([6.0 4.0; 4.0 8.0] ./ 4)); logscale = 0),
                (q = m(PointMass([-1.0, 2.0]), PointMass([7.0 -1.0; -1.0 9.0])),) => MvNormalMeanPrecision([-1.0, 2.0], [7.0 -1.0; -1.0 9.0]),
                (q = m(MvNormalMeanCovariance([1.0, 2.0], [3.0 2.0; 2.0 4.0]), PointMass([2.0 0.0; 0.0 2.0])),) => MvNormalMeanPrecision([1.0, 2.0], [2.0 0.0; 0.0 2.0]),
                (m = NamedTuple{(other,)}((PointMass([1.0, 3.0]),)), q = (Λ = PointMass([2.0 0.0; 0.0 2.0]),)) => MvNormalMeanPrecision([1.0, 3.0], [2.0 0.0; 0.0 2.0]),
                (m = NamedTuple{(other,)}((MvNormalMeanCovariance([2.0, 1.0], [3.0 2.0; 2.0 4.0]),)), q = (Λ = PointMass([6.0 4.0; 4.0 8.0] ./ 4),)) =>
                    MvNormalMeanCovariance([2.0, 1.0], [3.0 2.0; 2.0 4.0] + inv([6.0 4.0; 4.0 8.0] ./ 4)),
            ],
        )
        @test_message_update_rule(
            node = MvNormalMeanPrecision, target = target, float_types = (Float64,),
            cases = [
                (m = NamedTuple{(other,)}((MvNormalMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]),)), q = (Λ = Wishart(2.0, [6.0 4.0; 4.0 8.0] ./ 2.0),)) =>
                    MvNormalMeanCovariance([2.0, 1.0], [0.75 -0.375; -0.375 0.5625]),
                (m = NamedTuple{(other,)}((MvNormalMeanCovariance([0.0, 0.0], [7.0 -1.0; -1.0 9.0]),)), q = (Λ = Wishart(3.0, [12.0 -2.0; -2.0 7.0] ./ 3.0),)) =>
                    MvNormalMeanCovariance([0.0, 0.0], [567 / 80 -39 / 40; -39 / 40 183 / 20]),
                (q = m(MvNormalMeanCovariance([1.0, 2.0], [3.0 2.0; 2.0 4.0]), Wishart(4.0, [1.0 0.0; 0.0 1.0] ./ 4.0)),) => MvNormalMeanPrecision([1.0, 2.0], [1.0 0.0; 0.0 1.0]),
            ],
        )
    end
end

@testitem "rules:MvNormalMeanPrecision:Λ" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    using MessagePassingRulesBase: RuleContext
    using MatrixCorrectionTools: ReplaceZeroDiagonalEntries
    using FastCholesky: cholinv
    import ExponentialFamily: WishartFast

    # WishartFast(d + 2, E[(out - μ)(out - μ)ᵀ]).
    @test_message_update_rule(
        node = MvNormalMeanPrecision, target = :Λ,
        cases = [
            (q = (out = PointMass([1.0, 2.0]), μ = MvNormalMeanPrecision([3.0, 5.0], [3.0 2.0; 2.0 4.0])),) => WishartFast(4.0, cholinv([75 / 73 -46 / 73; -46 / 73 36 / 73])),
            (q = (out = MvNormalMeanCovariance([1.0, 2.0], [3.0 2.0; 2.0 4.0]), μ = MvNormalMeanCovariance([3.0, 5.0], [3.0 2.0; 2.0 4.0])),) => WishartFast(4.0, cholinv([17 / 70 -1 / 7; -1 / 7 1 / 7])),
            (clusters = ((:out, :μ) => MvNormalMeanCovariance(ones(4), [1.0 0 0 0; 0 1.0 0 0; 0 0 1.0 0; 0 0 0 1.0]),),) => WishartFast(4.0, [2.0 0.0; 0.0 2.0]),
        ],
    )

    # The scale matrix goes through `ctx.matrix_correction`: a zero on the diagonal, from two
    # point masses agreeing in a coordinate, is replaced by the strategy's value.
    degenerate = (q = (out = PointMass([1.0, 2.0]), μ = PointMass([1.0, 5.0])),)
    @test call_message_update_rule(MvNormalMeanPrecision, :Λ; degenerate...) == WishartFast(4.0, [0.0 0.0; 0.0 9.0])
    @test call_message_update_rule(MvNormalMeanPrecision, :Λ; degenerate..., ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(1.0e-3))) ==
        WishartFast(4.0, [1.0e-3 0.0; 0.0 9.0])
end

@testitem "rules:MvNormalMeanPrecision:marginals" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions

    I2 = [1.0 0.0; 0.0 1.0]
    # Two messages with precision I and E[Λ] = I/2 couple them.
    joint = MvNormalWeightedMeanPrecision([1.0, 2.0, 3.0, 4.0], [1.5 0 -0.5 0; 0 1.5 0 -0.5; -0.5 0 1.5 0; 0 -0.5 0 1.5])
    @test_marginal_update_rule(
        node = MvNormalMeanPrecision, target = (:out, :μ),
        cases = [
            (m = (out = MvNormalWeightedMeanPrecision([1.0, 2.0], I2), μ = MvNormalWeightedMeanPrecision([3.0, 4.0], I2)), q = (Λ = PointMass(I2 / 2),)) => joint,
            (m = (out = PointMass([1.0, 1.0]), μ = MvNormalWeightedMeanPrecision([3.0, 4.0], I2)), q = (Λ = PointMass(I2 / 2),)) =>
                FactorizedCluster((:out,) => PointMass([1.0, 1.0]), (:μ,) => MvNormalWeightedMeanPrecision([3.5, 4.5], 1.5 * I2)),
            (m = (out = MvNormalWeightedMeanPrecision([3.0, 4.0], I2), μ = PointMass([1.0, 1.0])), q = (Λ = PointMass(I2 / 2),)) =>
                FactorizedCluster((:out,) => MvNormalWeightedMeanPrecision([3.5, 4.5], 1.5 * I2), (:μ,) => PointMass([1.0, 1.0])),
        ],
    )
    @test_marginal_update_rule(
        node = MvNormalMeanPrecision, target = (:out, :μ), float_types = (Float64,),
        cases = [
            (m = (out = MvNormalWeightedMeanPrecision([1.0, 2.0], I2), μ = MvNormalWeightedMeanPrecision([3.0, 4.0], I2)), q = (Λ = Wishart(2.0, I2 / 4),)) => joint,
        ],
    )
    @test_marginal_update_rule(
        node = MvNormalMeanPrecision, target = (:out, :μ, :Λ),
        cases = [
            (m = (out = MvNormalWeightedMeanPrecision([1.0, 2.0], I2), μ = MvNormalWeightedMeanPrecision([3.0, 4.0], I2), Λ = PointMass(I2 / 2)),) =>
                FactorizedCluster((:out, :μ) => joint, (:Λ,) => PointMass(I2 / 2)),
            (m = (out = MvNormalWeightedMeanPrecision([1.0, 2.0], I2), μ = PointMass([1.0, 1.0]), Λ = PointMass(I2 / 2)),) =>
                FactorizedCluster((:out,) => MvNormalWeightedMeanPrecision([1.5, 2.5], 1.5 * I2), (:μ,) => PointMass([1.0, 1.0]), (:Λ,) => PointMass(I2 / 2)),
            (m = (out = PointMass([1.0, 1.0]), μ = MvNormalWeightedMeanPrecision([3.0, 4.0], I2), Λ = PointMass(I2 / 2)),) =>
                FactorizedCluster((:out,) => PointMass([1.0, 1.0]), (:μ,) => MvNormalWeightedMeanPrecision([3.5, 4.5], 1.5 * I2), (:Λ,) => PointMass(I2 / 2)),
        ],
    )
end

@testitem "rules:MvNormalMeanPrecision:average-energy" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    using StatsFuns: log2π
    using LinearAlgebra: logdet

    I2 = [1.0 0.0; 0.0 1.0]
    # In each representation of q_μ, with a Wishart q_Λ.
    for q_μ in (MvNormalMeanPrecision([1.0, 1.0], I2), MvNormalMeanCovariance([1.0, 1.0], I2), MvNormalWeightedMeanPrecision([1.0, 1.0], I2))
        @test call_average_energy(MvNormalMeanPrecision; q = (out = PointMass([1.0, 1.0]), μ = q_μ, Λ = Wishart(3, 2 * I2))) ≈ 6.721945550750932
    end
    # By hand: (d log 2π - E[log |Λ|] + tr(E[Λ] S)) / 2, the Wishart shortcut against the mean.
    q_Λ = Wishart(4.0, [1.0 0.2; 0.2 0.5])
    q = (out = MvNormalMeanCovariance([0.5, 1.0], [1.0 0.2; 0.2 0.5]), μ = MvNormalMeanCovariance([1.0, 1.0], I2), Λ = q_Λ)
    S = [1.0 0.2; 0.2 0.5] + I2 + [0.25 0.0; 0.0 0.0]
    @test call_average_energy(MvNormalMeanPrecision; q) ≈ (2 * log2π - mean(logdet, q_Λ) + sum(mean(q_Λ) .* S)) / 2
    @test call_average_energy(MvNormalMeanPrecision; q = (q..., Λ = PointMass(mean(q_Λ)))) ≈ (2 * log2π - logdet(mean(q_Λ)) + sum(mean(q_Λ) .* S)) / 2
    # The joint's blocks give S = V₁₁ + V₂₂ - V₁₂ - V₂₁ = 2I, so tr(E[Λ] S) = tr(I/2 · 2I) = 2.
    joint = MvNormalMeanCovariance([1.0, 1.0, 1.0, 1.0], [1.0 0 0 0; 0 1.0 0 0; 0 0 1.0 0; 0 0 0 1.0])
    @test call_average_energy(MvNormalMeanPrecision; q = (Λ = PointMass(I2 / 2),), clusters = ((:out, :μ) => joint,)) ≈ (2 * log2π - logdet(I2 / 2) + 2.0) / 2
end
