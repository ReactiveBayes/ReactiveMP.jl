# MvNormalWishart: known parameters, with ν ≥ d, as a Wishart at d = 2 needs.

@testitem "rules:MvNormalWishart:out" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions

    @test_message_update_rule(
        node = MvNormalWishart, target = :out,
        cases = [
            (q = (μ = PointMass([1.0, 2.0]), W = PointMass([1.0 0.0; 0.0 1.0]), λ = PointMass(1.0), ν = PointMass(3.0)),) =>
                MvNormalWishart([1.0, 2.0], [1.0 0.0; 0.0 1.0], 1.0, 3.0),
            (q = (μ = PointMass([0.5, -1.0]), W = PointMass([2.0 0.3; 0.3 1.5]), λ = PointMass(2.5), ν = PointMass(4.0)),) =>
                MvNormalWishart([0.5, -1.0], [2.0 0.3; 0.3 1.5], 2.5, 4.0),
        ],
    )
end
