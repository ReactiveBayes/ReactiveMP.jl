# MatrixNormal: tables of cases, with the expected scale matrices written out as derived,
# and the energy against -logpdf for point masses and a hand-derived second-moment case.

@testitem "rules:MatrixNormal:out-M" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions, LinearAlgebra

    I2, I3 = Matrix(1.0I, 2, 2), Matrix(1.0I, 3, 3)
    D3 = [2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0]
    for (tgt, src) in ((:out, :M), (:M, :out))
        m(x, U, V) = (m = NamedTuple{(src, :U, :V)}((x, U, V)),)
        q(x, U, V) = (q = NamedTuple{(src, :U, :V)}((x, U, V)),)
        @test_message_update_rule(
            node = MatrixNormal, target = tgt,
            cases = [
                m(PointMass([1.0 2.0; 3.0 4.0]), PointMass([2.0 0.0; 0.0 3.0]), PointMass([1.0 0.0; 0.0 4.0])) => MatrixNormal([1.0 2.0; 3.0 4.0], [2.0 0.0; 0.0 3.0], [1.0 0.0; 0.0 4.0]),
                m(PointMass([0.5 1.0; 2.0 3.0; 4.0 5.0]), PointMass(D3), PointMass([1.0 0.0; 0.0 2.0])) => MatrixNormal([0.5 1.0; 2.0 3.0; 4.0 5.0], D3, [1.0 0.0; 0.0 2.0]),
                m(PointMass(zeros(2, 3)), PointMass(I2), PointMass(I3)) => MatrixNormal(zeros(2, 3), I2, I3),
                m(MatrixNormal([1.0 2.0; 3.0 4.0], [0.5 0.0; 0.0 0.5], [1.0 0.0; 0.0 4.0]), PointMass([2.0 0.5; 0.5 3.0]), PointMass([1.0 0.0; 0.0 2.0])) =>
                    MvNormalMeanCovariance(vec([1.0 2.0; 3.0 4.0]), kron([1.0 0.0; 0.0 2.0], [2.0 0.5; 0.5 3.0]) + kron([1.0 0.0; 0.0 4.0], [0.5 0.0; 0.0 0.5])),
                m(MatrixNormal(zeros(3, 2), I3, I2), PointMass(D3), PointMass([1.0 0.0; 0.0 2.0])) =>
                    MvNormalMeanCovariance(vec(zeros(3, 2)), kron([1.0 0.0; 0.0 2.0], D3) + kron(I2, I3)),
                m(PointMass([1.0 2.0; 3.0 4.0]), InverseWishart(5.0, [2.0 0.0; 0.0 2.0]), PointMass([1.0 0.0; 0.0 2.0])) =>
                    MatrixTDist(4.0, [1.0 2.0; 3.0 4.0], [2.0 0.0; 0.0 2.0], [1.0 0.0; 0.0 2.0]),
                m(PointMass(zeros(3, 2)), InverseWishart(7.0, D3), PointMass([1.0 0.5; 0.5 2.0])) => MatrixTDist(5.0, zeros(3, 2), D3, [1.0 0.5; 0.5 2.0]),
                m(PointMass([1.0 2.0; 3.0 4.0]), PointMass([2.0 0.5; 0.5 3.0]), InverseWishart(5.0, I2)) => MatrixTDist(4.0, [1.0 2.0; 3.0 4.0], [2.0 0.5; 0.5 3.0], I2),
                m(PointMass(zeros(3, 2)), PointMass(D3), InverseWishart(6.0, [1.0 0.5; 0.5 2.0])) => MatrixTDist(5.0, zeros(3, 2), D3, [1.0 0.5; 0.5 2.0]),
                # E[U⁻¹]⁻¹ = Ψ/ν for an InverseWishart(ν, Ψ).
                q(MatrixNormal([1.0 2.0; 3.0 4.0], [2.0 0.5; 0.5 3.0], [1.0 0.0; 0.0 2.0]), InverseWishart(5.0, [2.0 0.0; 0.0 2.0]), InverseWishart(4.0, I2)) =>
                    MatrixNormal([1.0 2.0; 3.0 4.0], [2 / 5 0.0; 0.0 2 / 5], [1 / 4 0.0; 0.0 1 / 4]),
                # A known `src`.
                q(PointMass([1.0 2.0; 3.0 4.0]), InverseWishart(5.0, [2.0 0.0; 0.0 2.0]), PointMass([1.0 0.0; 0.0 2.0])) =>
                    MatrixNormal([1.0 2.0; 3.0 4.0], [2 / 5 0.0; 0.0 2 / 5], [1.0 0.0; 0.0 2.0]),
            ],
        )
    end
end

@testitem "rules:MatrixNormal:U-V" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions, LinearAlgebra
    using FastCholesky: cholinv
    import ExponentialFamily: InverseWishartFast

    X, M = [1.0 2.0; 3.0 4.0; 5.0 6.0], [0.5 1.0; 2.0 3.0; 4.0 5.0]
    D = X - M
    U3, V2 = [0.5 0.0 0.0; 0.0 0.5 0.0; 0.0 0.0 0.5], [1.0 0.0; 0.0 0.5]
    q_V, q_U = InverseWishart(5.0, [2.0 0.0; 0.0 2.0]), InverseWishart(6.0, [2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0])
    B, A = mean(cholinv, q_V), mean(cholinv, q_U)
    # Degrees of freedom p - n - 1 = -2 and n - p - 1 = 0 for a 3×2 `out`, and a looser
    # tolerance for the conversions through an InverseWishart.
    @test_message_update_rule(
        node = MatrixNormal, target = :U, atol = 1.0e-3,
        cases = [
            (m = (out = PointMass(X), M = PointMass(M), V = PointMass([1.0 0.0; 0.0 2.0])),) => InverseWishartFast(-2.0, D * cholinv([1.0 0.0; 0.0 2.0]) * D'),
            (m = (out = PointMass([2.0 1.0; -1.0 0.5]), M = PointMass([1.0 0.0; 0.0 1.0]), V = PointMass([2.0 -0.5; -0.5 1.0])),) =>
                InverseWishartFast(-1.0, [1.0 1.0; -1.0 -0.5] * cholinv([2.0 -0.5; -0.5 1.0]) * [1.0 1.0; -1.0 -0.5]'),
            (q = (out = PointMass(X), M = PointMass(M), V = q_V),) => InverseWishartFast(-2.0, D * B * D'),
            (q = (out = MatrixNormal(X, U3, V2), M = PointMass(M), V = q_V),) => InverseWishartFast(-2.0, D * B * D' + tr(B * V2) * U3),
            (q = (out = PointMass(X), M = MatrixNormal(M, U3, V2), V = q_V),) => InverseWishartFast(-2.0, D * B * D' + tr(B * V2) * U3),
        ],
    )
    @test_message_update_rule(
        node = MatrixNormal, target = :V, atol = 1.0e-3,
        cases = [
            (m = (out = PointMass(X), M = PointMass(M), U = PointMass(U3)),) => InverseWishartFast(0.0, D' * cholinv(U3) * D),
            (q = (out = PointMass(X), M = PointMass(M), U = q_U),) => InverseWishartFast(0.0, D' * A * D),
            (q = (out = MatrixNormal(X, U3, V2), M = PointMass(M), U = q_U),) => InverseWishartFast(0.0, D' * A * D + tr(A * U3) * V2),
            (q = (out = PointMass(X), M = MatrixNormal(M, U3, V2), U = q_U),) => InverseWishartFast(0.0, D' * A * D + tr(A * U3) * V2),
        ],
    )
end

@testitem "rules:MatrixNormal:marginals-energy" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions, LinearAlgebra

    @test_marginal_update_rule(
        node = MatrixNormal, target = (:out, :M, :U, :V),
        cases = [
            (m = (out = PointMass([1.0 2.0; 3.0 4.0]), M = PointMass([0.5 1.0; 1.5 2.0]), U = PointMass([2.0 0.0; 0.0 3.0]), V = PointMass([1.0 0.0; 0.0 4.0])),) =>
                FactorizedCluster(
                (:out,) => PointMass([1.0 2.0; 3.0 4.0]), (:M,) => PointMass([0.5 1.0; 1.5 2.0]), (:U,) => PointMass([2.0 0.0; 0.0 3.0]), (:V,) => PointMass([1.0 0.0; 0.0 4.0]),
            ),
        ],
    )
    # All point masses: the energy is -logpdf.
    X, M, U, V = [1.0 2.0; 3.0 4.0; 5.0 6.0], [0.5 1.0; 2.0 3.0; 4.0 5.0], [2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0], [1.0 0.5; 0.5 2.0]
    X2, M2, U2, V2 = [1.0 2.0; 3.0 4.0], [0.5 1.0; 1.5 2.0], [2.0 0.3; 0.3 1.5], [1.0 0.2; 0.2 2.0]
    # A MatrixNormal q(out) adds tr(V⁻¹ V_out) U_out to the scatter.
    U_out, V_out = [0.5 0.0; 0.0 0.5], [1.0 0.0; 0.0 0.5]
    D = X2 - M2
    Ψ = D * inv(V2) * D' + tr(inv(V2) * V_out) * U_out
    @test_average_energy(
        node = MatrixNormal,
        cases = [
            (q = (out = PointMass(X), M = PointMass(M), U = PointMass(U), V = PointMass(V)),) => -logpdf(MatrixNormal(M, U, V), X),
            (q = (out = PointMass(X2), M = PointMass(M2), U = PointMass(U2), V = PointMass(V2)),) => -logpdf(MatrixNormal(M2, U2, V2), X2),
            (q = (out = MatrixNormal(X2, U_out, V_out), M = PointMass(M2), U = PointMass(U2), V = PointMass(V2)),) =>
                (2 * logdet(U2) + 2 * logdet(V2) + 4 * log(2π) + tr(inv(U2) * Ψ)) / 2,
        ],
    )
end
