# InverseWishart: tables of cases, and hand-derived ones for a `q_S` and a Wishart-family
# `out` message.

@testitem "rules:InverseWishart:out" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions
    import ExponentialFamily: InverseWishartFast

    I2, I3 = [1.0 0.0; 0.0 1.0], [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    point_masses = [(2.0, I2), (3.0, [10.0 -1.0; -1.0 3.0]), (4.0, I3)]
    @test_message_update_rule(
        node = InverseWishart, target = :out,
        cases = [
            [(m = (ν = PointMass(ν), S = PointMass(S)),) => InverseWishartFast(ν, S) for (ν, S) in point_masses];
            [(m = (ν = PointMass(ν),), q = (S = PointMass(S),)) => InverseWishartFast(ν, S) for (ν, S) in point_masses];
            [(q = (ν = PointMass(ν),), m = (S = PointMass(S),)) => InverseWishartFast(ν, S) for (ν, S) in point_masses];
            [(q = (ν = PointMass(ν), S = PointMass(S)),) => InverseWishartFast(ν, S) for (ν, S) in point_masses];
            # The scale is E[S] = ν₀ V for S ~ Wishart(ν₀, V).
            (m = (ν = PointMass(3.0),), q = (S = Wishart(4.0, [1.0 0.5; 0.5 2.0]),)) => InverseWishartFast(3.0, [4.0 2.0; 2.0 8.0])
        ],
    )
end

@testitem "rules:InverseWishart:marginals-energy" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    import ExponentialFamily: InverseWishartFast

    I2 = [1.0 0.0; 0.0 1.0]
    A, B = [9.0 -2.0 1.0; -2.0 5.0 -2.0; 1.0 -2.0 11.0], [11.0 -2.0 1.0; -2.0 5.0 -2.0; 1.0 -2.0 9.0]
    # Degrees of freedom ν₁ + ν₂ + d + 1, scale Ψ₁ + Ψ₂.
    @test_marginal_update_rule(
        node = InverseWishart, target = (:out, :ν, :S),
        cases = [
            (m = (out = InverseWishartFast(3.0, [3.0 -1.0; -1.0 4.0]), ν = PointMass(2.0), S = PointMass(I2)),) =>
                FactorizedCluster((:out,) => InverseWishart(8.0, [4.0 -1.0; -1.0 5.0]), (:ν,) => PointMass(2.0), (:S,) => PointMass(I2)),
            (m = (out = InverseWishartFast(4.0, A), ν = PointMass(3.0), S = PointMass(B)),) =>
                FactorizedCluster((:out,) => InverseWishart(11.0, [20.0 -4.0 2.0; -4.0 10.0 -4.0; 2.0 -4.0 20.0]), (:ν,) => PointMass(3.0), (:S,) => PointMass(B)),
            (m = (out = InverseWishart(3.0, [3.0 -1.0; -1.0 4.0]), ν = PointMass(2.0), S = PointMass(I2)),) =>
                FactorizedCluster((:out,) => InverseWishart(8.0, [4.0 -1.0; -1.0 5.0]), (:ν,) => PointMass(2.0), (:S,) => PointMass(I2)),
        ],
    )
    S = [
        4.3082195553088445 0.4573472347695425 -2.748089173206861;
        0.4573472347695425 0.0954087613417567 -0.29586598556052124;
        -2.748089173206861 -0.29586598556052124 2.9875706318257538
    ]
    # ExponentialFamily's E[log |out|] for an InverseWishart is a Float64 whatever its inputs,
    # and its E[out⁻¹] fails for BigFloat (both ExponentialFamily.jl#322), so the energy is
    # checked in Float64 only.
    @test_average_energy(
        node = InverseWishart, float_types = (Float64,),
        cases = [
            (q = (out = InverseWishart(2.0, [2.0 0.0; 0.0 2.0]), ν = PointMass(2.0), S = PointMass([2.0 0.0; 0.0 2.0])),) => 9.496544113156787,
            (q = (out = InverseWishart(4.0, S), ν = PointMass(4.0), S = PointMass(S)),) => 1.1299587008097587,
        ],
    )
end
