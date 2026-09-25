# MatrixNormalWishart: tables of cases, and the energy in closed form, split into its two
# factors, the Wishart one computed by the Wishart node's own energy.

@testitem "rules:MatrixNormalWishart:out" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions

    @test_message_update_rule(
        node = MatrixNormalWishart, target = :out,
        cases = [
            (q = (M = PointMass([1.0 2.0; 3.0 4.0]), U = PointMass([2.0 0.0; 0.0 2.0]), V = PointMass([1.0 0.0; 0.0 1.0]), ν = PointMass(5.0)),) =>
                MatrixNormalWishart([1.0 2.0; 3.0 4.0], [2.0 0.0; 0.0 2.0], [1.0 0.0; 0.0 1.0], 5.0),
            (q = (M = PointMass([0.5 1.0; 2.0 3.0; 4.0 5.0]), U = PointMass([2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0]), V = PointMass([1.0 0.0; 0.0 2.0]), ν = PointMass(4.0)),) =>
                MatrixNormalWishart([0.5 1.0; 2.0 3.0; 4.0 5.0], [2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0], [1.0 0.0; 0.0 2.0], 4.0),
        ],
    )
end

@testitem "rules:MatrixNormalWishart:energy" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions, LinearAlgebra
    using StatsFuns: log2π

    M, U, V, ν = [1.0 2.0; 3.0 4.0], [2.0 0.3; 0.3 1.5], [1.0 0.2; 0.2 1.5], 5.0
    Mq, Uq, Vq, νq = [0.5 1.0; 1.5 2.0], [1.5 0.2; 0.2 1.0], [1.0 0.1; 0.1 1.2], 6.0
    q_Y = Wishart(νq, Vq)
    n, p = size(Mq)
    D = Mq - M
    # E[-log MatrixNormal(X; M, U, Y⁻¹)] = (n p log 2π + p log |U| - n E[log |Y|]
    # + tr(U⁻¹ (D E[Y] Dᵀ + p Uq))) / 2, and E[-log Wishart(Y; ν, V)] is the Wishart node's energy.
    matrix_normal_part = (n * p * log2π + p * logdet(U) - n * mean(logdet, q_Y) + tr(inv(U) * (D * mean(q_Y) * D' + p * Uq))) / 2
    wishart_part = call_average_energy(Wishart; q = (out = q_Y, ν = PointMass(ν), S = PointMass(V)))
    @test_average_energy(
        node = MatrixNormalWishart,
        cases = [
            (q = (out = MatrixNormalWishart(Mq, Uq, Vq, νq), M = PointMass(M), U = PointMass(U), V = PointMass(V), ν = PointMass(ν)),) => matrix_normal_part + wishart_part,
        ],
    )
end
