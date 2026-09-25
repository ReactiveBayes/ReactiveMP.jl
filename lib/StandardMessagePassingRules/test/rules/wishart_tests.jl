# Wishart: tables of cases, hand-derived ones for a `q_S` that is not a point mass, and
# `public_equivalent`.

@testitem "rules:Wishart:out" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions
    using FastCholesky: cholinv
    import ExponentialFamily: WishartFast

    I2, I3 = [1.0 0.0; 0.0 1.0], [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    S = [10.0 -1.0; -1.0 3.0]
    point_masses = [(2.0, I2), (3.0, S), (4.0, I3)]
    @test_message_update_rule(
        node = Wishart, target = :out,
        cases = [
            [(m = (ν = PointMass(ν), S = PointMass(S)),) => WishartFast(ν, cholinv(S)) for (ν, S) in point_masses];
            [(m = (ν = PointMass(ν),), q = (S = PointMass(S),)) => WishartFast(ν, cholinv(S)) for (ν, S) in point_masses];
            [(q = (ν = PointMass(ν),), m = (S = PointMass(S),)) => WishartFast(ν, cholinv(S)) for (ν, S) in point_masses];
            [(q = (ν = PointMass(ν), S = PointMass(S)),) => WishartFast(ν, cholinv(S)) for (ν, S) in point_masses];
            # The inverse scale is E[S⁻¹] = ν₀ Ψ⁻¹ for S ~ InverseWishart(ν₀, Ψ), not E[S]⁻¹.
            (m = (ν = PointMass(3.0),), q = (S = InverseWishart(5.0, [2.0 0.0; 0.0 4.0]),)) => WishartFast(3.0, [2.5 0.0; 0.0 1.25]);
            (q = (ν = PointMass(3.0), S = InverseWishart(5.0, [2.0 0.0; 0.0 4.0])),) => WishartFast(3.0, [2.5 0.0; 0.0 1.25])
        ],
    )
end

@testitem "rules:Wishart:marginals-energy" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    using FastCholesky: cholinv
    import ExponentialFamily: WishartFast

    I2 = [1.0 0.0; 0.0 1.0]
    B = [11.0 -2.0 1.0; -2.0 5.0 -2.0; 1.0 -2.0 9.0]
    @test_marginal_update_rule(
        node = Wishart, target = (:out, :ν, :S),
        cases = [
            (m = (out = WishartFast(3.0, cholinv([3.0 -1.0; -1.0 4.0])), ν = PointMass(2.0), S = PointMass(I2)),) =>
                FactorizedCluster((:out,) => Wishart(2.0, [14 / 19 -1 / 19; -1 / 19 15 / 19]), (:ν,) => PointMass(2.0), (:S,) => PointMass(I2)),
            (m = (out = WishartFast(7.0, cholinv([9.0 -2.0; -2.0 1.0])), ν = PointMass(4.0), S = PointMass([4.0 -2.0; -2.0 4.0])),) =>
                FactorizedCluster((:out,) => Wishart(8.0, [128 / 49 -34 / 49; -34 / 49 32 / 49]), (:ν,) => PointMass(4.0), (:S,) => PointMass([4.0 -2.0; -2.0 4.0])),
            (m = (out = WishartFast(4.0, cholinv([9.0 -2.0 1.0; -2.0 5.0 -2.0; 1.0 -2.0 11.0])), ν = PointMass(3.0), S = PointMass(B)),) =>
                FactorizedCluster(
                (:out,) => Wishart(3.0, [2092 / 423 -1.0 211 / 423; -1.0 5 / 2 -1.0; 211 / 423 -1.0 2092 / 423]), (:ν,) => PointMass(3.0), (:S,) => PointMass(B),
            ),
        ],
    )
    S = [
        4.3082195553088445 0.4573472347695425 -2.748089173206861;
        0.4573472347695425 0.0954087613417567 -0.29586598556052124;
        -2.748089173206861 -0.29586598556052124 2.9875706318257538
    ]
    @test_average_energy(
        node = Wishart,
        cases = [
            (q = (out = Wishart(2.0, [2.0 0.0; 0.0 2.0]), ν = PointMass(2.0), S = PointMass([2.0 0.0; 0.0 2.0])),) => 6.033250123747594,
            (q = (out = Wishart(4.0, S), ν = PointMass(4.0), S = PointMass(S)),) => 8.97595944423116,
        ],
    )
end

@testitem "rules:Wishart:public-equivalent" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, Distributions
    using FastCholesky: cholinv
    import ExponentialFamily: WishartFast, InverseWishartFast

    S = [1.0 0.1; 0.1 2.0]
    wishart = public_equivalent(WishartFast(5.0, cholinv(S)))
    @test wishart isa Wishart
    @test params(wishart)[1] == 5.0 && Matrix(params(wishart)[2]) ≈ S
    inverse = public_equivalent(InverseWishartFast(5.0, S))
    @test inverse isa InverseWishart
    @test params(inverse)[1] == 5.0 && Matrix(params(inverse)[2]) ≈ S
    # A public type is left as it is.
    @test public_equivalent(wishart) === wishart
end
