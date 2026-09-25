# MvNormalWeightedMeanPrecision: tables of cases, a hand-derived marginal, and a node test,
# which checks the energy against MvNormalMeanPrecision's for the same density.

@testitem "rules:MvNormalWeightedMeanPrecision" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions

    I2 = [1.0 0.0; 0.0 1.0]
    @test_message_update_rule(
        node = MvNormalWeightedMeanPrecision, target = :out,
        cases = [
            (m = (ξ = PointMass([1.0, 3.0]), Λ = PointMass([3.0 2.0; 2.0 4.0])),) => MvNormalWeightedMeanPrecision([1.0, 3.0], [3.0 2.0; 2.0 4.0]),
            (q = (ξ = PointMass([-1.0, 2.0]), Λ = PointMass([7.0 -1.0; -1.0 9.0])),) => MvNormalWeightedMeanPrecision([-1.0, 2.0], [7.0 -1.0; -1.0 9.0]),
            (q = (ξ = MvNormalMeanCovariance([1.0, 2.0], [3.0 2.0; 2.0 4.0]), Λ = PointMass(2 * I2)),) => MvNormalWeightedMeanPrecision([1.0, 2.0], 2 * I2),
        ],
    )
    # out's block: its message times N(Λ⁻¹ξ, Λ⁻¹), weighted means and precisions adding up.
    @test_marginal_update_rule(
        node = MvNormalWeightedMeanPrecision, target = (:out, :ξ, :Λ),
        cases = [
            (m = (out = MvNormalWeightedMeanPrecision([1.0, 2.0], I2), ξ = PointMass([0.5, 0.5]), Λ = PointMass(2 * I2)),) =>
                FactorizedCluster((:out,) => MvNormalWeightedMeanPrecision([1.5, 2.5], 3 * I2), (:ξ,) => PointMass([0.5, 0.5]), (:Λ,) => PointMass(2 * I2)),
        ],
    )
    # The energy of N(x; Λ⁻¹ξ, Λ⁻¹) is MvNormalMeanPrecision's for mean Λ⁻¹ξ, when ξ and Λ are
    # point masses.
    Λ = [2.0 0.3; 0.3 1.5]
    ξ = [0.5, -1.0]
    for q_out in (PointMass([0.2, 0.4]), MvNormalMeanCovariance([0.2, 0.4], [1.0 0.1; 0.1 0.7]))
        @test getresult(call_average_energy(MvNormalWeightedMeanPrecision; q = (out = q_out, ξ = PointMass(ξ), Λ = PointMass(Λ)))) ≈
            getresult(call_average_energy(MvNormalMeanPrecision; q = (out = q_out, μ = PointMass(Λ \ ξ), Λ = PointMass(Λ))))
    end
end
