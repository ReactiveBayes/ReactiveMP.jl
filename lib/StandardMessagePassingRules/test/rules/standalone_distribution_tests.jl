# `out ~ d` for a distribution value `d`: `d` arrives as a
# constant's point mass.

@testitem "rules:StandaloneDistribution" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions

    # The message towards `out` is the distribution itself, whatever its family.
    d = Beta(4.0, 8.0)
    truncated_normal = Truncated(Normal(0.5, 1.0), 0.0, 1.0)
    @test getresult(call_message_update_rule(StandaloneDistribution, :out; q = (distribution = PointMass(d),))) === d
    @test getresult(call_message_update_rule(StandaloneDistribution, :out; q = (distribution = PointMass(truncated_normal),))) === truncated_normal

    # The average energy is the cross entropy E_q[-log d(out)] = KL(q ‖ d) + H(q), so the node's
    # free-energy term, the energy less H(q), is KL(q ‖ d).
    q = Beta(6.0, 9.0)
    energy = getresult(call_average_energy(StandaloneDistribution; q = (out = q, distribution = PointMass(d))))
    @test energy ≈ kldivergence(q, d) + entropy(q)
    @test energy - entropy(q) ≈ kldivergence(q, d)
    # Against the integral.
    @test energy ≈ -sum(x -> logpdf(d, x) * pdf(q, x), range(1.0e-6, 1 - 1.0e-6; length = 200_001)) * (1 - 2.0e-6) / 200_000 atol = 1.0e-4
end
