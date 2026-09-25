# MvNormalGamma: tables of cases, a hand-derived one for an uncertain μ, and the energy checked
# against the entropy it must equal when the prior is q(out) itself.

@testitem "rules:MvNormalGamma:out" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions

    μ0, Λ0 = [0.5, -1.0], [2.0 0.3; 0.3 1.5]
    @test_message_update_rule(
        node = MvNormalGamma, target = :out,
        cases = [
            (m = (μ = PointMass(μ0), Λ = PointMass(Λ0), α = PointMass(2.0), β = PointMass(3.0)),) => MvNormalGamma(μ0, Λ0, 2.0, 3.0),
            (q = (μ = PointMass([0.0]), Λ = PointMass(fill(1.0, 1, 1)), α = PointMass(1.0), β = PointMass(1.0)),) => MvNormalGamma([0.0], fill(1.0, 1, 1), 1.0, 1.0),
            # rate 3 + tr(E[Λ] Cov μ)/2 = 3 + (2·1 + 1·2)/2; E[α] = 2 and E[β] = 3.
            (q = (μ = MvNormalMeanCovariance([0.0, 1.0], [1.0 0.0; 0.0 2.0]), Λ = PointMass([2.0 0.0; 0.0 1.0]), α = GammaShapeRate(4.0, 2.0), β = PointMass(3.0)),) =>
                MvNormalGamma([0.0, 1.0], [2.0 0.0; 0.0 1.0], 2.0, 5.0),
        ],
    )
end

@testitem "rules:MvNormalGamma:energy" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions

    for q in (MvNormalGamma([0.3], fill(1.5, 1, 1), 4.2, 2.5), MvNormalGamma([0.5, -1.0], [2.0 0.3; 0.3 1.5], 4.5, 2.2))
        μ, Λ, α, β = params(q)
        # The cross-entropy of q against itself is its entropy, which ExponentialFamily computes
        # independently; against another prior it is larger.
        @test_average_energy(
            node = MvNormalGamma,
            cases = [(q = (out = q, μ = PointMass(μ), Λ = PointMass(Λ), α = PointMass(α), β = PointMass(β)),) => entropy(q)],
        )
        other = call_average_energy(MvNormalGamma; q = (out = q, μ = PointMass(μ .+ 1), Λ = PointMass(2Λ), α = PointMass(α / 2), β = PointMass(β + 1)))
        @test isfinite(other) && other > entropy(q)
    end
end
