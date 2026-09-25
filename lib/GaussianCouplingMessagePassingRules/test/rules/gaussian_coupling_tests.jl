# GaussianCoupling. The messages are the GaBP messages of arXiv:0810.1119 with
# a = -A[i, j]: ξ = a ⋅ μ, precision -a² ⋅ v, improper by design. The cross-checks against
# NormalMeanPrecision's rules are written as their closed forms, N(mean(m), var(m) + 1/w).

@testitem "rules:GaussianCoupling:in-out" tags = [:rules] begin
    using GaussianCouplingMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase

    for (target, other) in ((:in, :out), (:out, :in))
        m(value) = NamedTuple{(other,)}((value,))
        @test_message_update_rule(
            node = GaussianCoupling, target = target,
            cases = [
                (m = m(NormalMeanVariance(0.0, 1.0)), q = (a = PointMass(1.0),)) => NormalWeightedMeanPrecision(0.0, -1.0),
                (m = m(NormalMeanVariance(2.0, 3.0)), q = (a = PointMass(1.0),)) => NormalWeightedMeanPrecision(2.0, -3.0),
                # A negative coupling, the actual usage, since a = -A[i, j].
                (m = m(NormalMeanVariance(2.0, 3.0)), q = (a = PointMass(-0.5),)) => NormalWeightedMeanPrecision(-1.0, -0.75),
                (m = m(NormalMeanPrecision(-1.0, 4.0)), q = (a = PointMass(-2.0),)) => NormalWeightedMeanPrecision(2.0, -1.0),
                (m = m(NormalWeightedMeanPrecision(1.0, 2.0)), q = (a = PointMass(3.0),)) => NormalWeightedMeanPrecision(1.5, -4.5),
                # A zero coupling decouples the two variables.
                (m = m(NormalMeanVariance(2.0, 3.0)), q = (a = PointMass(0.0),)) => NormalWeightedMeanPrecision(0.0, 0.0),
            ],
        )
    end

    # The factor is symmetric in `out` and `in`.
    for (m, a) in ((NormalMeanVariance(1.5, 2.0), 0.75), (NormalMeanPrecision(-2.0, 3.0), -1.25))
        @test call_message_update_rule(GaussianCoupling, :out; m = (in = m,), q = (a = PointMass(a),)) ≈
            call_message_update_rule(GaussianCoupling, :in; m = (out = m,), q = (a = PointMass(a),))
    end

    # N(out; in, 1/w) ∝ exp(-w out²/2) exp(w out in) exp(-w in²/2): with a = w the coupling is the
    # normal's cross term, so tilting the incoming precision by w and multiplying by the message
    # NormalWeightedMeanPrecision(0, w) gives the normal's message, N(mean(m), var(m) + 1/w).
    for (ξ, w_other, w) in ((0.5, 2.0, 1.0), (-1.0, 4.0, 0.5)), (target, other) in ((:in, :out), (:out, :in))
        m = NormalWeightedMeanPrecision(ξ, w_other)
        tilted = NamedTuple{(other,)}((NormalWeightedMeanPrecision(ξ, w_other + w),))
        message = prod(GenericProd(), NormalWeightedMeanPrecision(0.0, w), call_message_update_rule(GaussianCoupling, target; m = tilted, q = (a = PointMass(w),)))
        @test all(mean_var(message) .≈ (mean(m), var(m) + 1 / w))
    end
end

@testitem "rules:GaussianCoupling:marginals" tags = [:rules] begin
    using GaussianCouplingMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase

    # q(out, in) ∝ m_out(out) m_in(in) exp(a out in): ξ = [ξ_out, ξ_in], W = [w_out -a; -a w_in].
    @test_marginal_update_rule(
        node = GaussianCoupling, target = (:out, :in),
        cases = [
            (m = (out = NormalWeightedMeanPrecision(1.0, 2.0), in = NormalWeightedMeanPrecision(2.0, 3.0)), q = (a = PointMass(1.0),)) => MvNormalWeightedMeanPrecision([1.0, 2.0], [2.0 -1.0; -1.0 3.0]),
            (m = (out = NormalWeightedMeanPrecision(1.0, 2.0), in = NormalWeightedMeanPrecision(2.0, 3.0)), q = (a = PointMass(-1.5),)) => MvNormalWeightedMeanPrecision([1.0, 2.0], [2.0 1.5; 1.5 3.0]),
            (m = (out = NormalMeanVariance(1.0, 0.5), in = NormalMeanVariance(-2.0, 0.25)), q = (a = PointMass(-1.5),)) => MvNormalWeightedMeanPrecision([2.0, -8.0], [2.0 1.5; 1.5 4.0]),
            (m = (out = NormalMeanPrecision(2.0, 4.0), in = NormalMeanPrecision(-1.0, 2.0)), q = (a = PointMass(0.5),)) => MvNormalWeightedMeanPrecision([8.0, -2.0], [4.0 -0.5; -0.5 2.0]),
            # A zero coupling leaves the joint block-diagonal.
            (m = (out = NormalMeanVariance(1.0, 0.5), in = NormalMeanVariance(-2.0, 0.25)), q = (a = PointMass(0.0),)) => MvNormalWeightedMeanPrecision([2.0, -8.0], [2.0 0.0; 0.0 4.0]),
        ],
    )

    # The joint is proper exactly when w_out ⋅ w_in > a², a local condition on the factor.
    smallest_eigenvalue(W) = (W[1, 1] + W[2, 2]) / 2 - sqrt(((W[1, 1] - W[2, 2]) / 2)^2 + W[1, 2]^2)
    for (w_out, w_in, a, proper) in ((4.0, 4.0, 1.0, true), (2.0, 2.0, 1.9, true), (2.0, 2.0, 2.0, false), (1.0, 1.0, 2.0, false))
        q = call_marginal_update_rule(GaussianCoupling, (:out, :in); m = (out = NormalWeightedMeanPrecision(0.5, w_out), in = NormalWeightedMeanPrecision(-0.5, w_in)), q = (a = PointMass(a),))
        W = convert(Matrix, precision(q))
        @test (smallest_eigenvalue(W) > sqrt(eps()) * maximum(abs, W)) === proper
        @test (w_out * w_in > abs2(a)) === proper
    end
end

@testitem "rules:GaussianCoupling:energy" tags = [:rules] begin
    using GaussianCouplingMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase

    # ⟨-log φ⟩ = -E[a] E[out ⋅ in] = -E[a] (V[1, 2] + m[1] m[2]).
    joint = MvNormalMeanCovariance([1.0, 2.0], [2.0 0.5; 0.5 3.0])
    @test_average_energy(
        node = GaussianCoupling,
        cases = [
            (clusters = ((:out, :in) => joint,), q = (a = PointMass(2.0),)) => -5.0,
            (clusters = ((:out, :in) => joint,), q = (a = PointMass(-2.0),)) => 5.0,     # linear in a
            (clusters = ((:out, :in) => joint,), q = (a = PointMass(0.0),)) => 0.0,
            # The covariance counts even with zero means.
            (clusters = ((:out, :in) => MvNormalMeanCovariance([0.0, 0.0], [1.0 -0.5; -0.5 1.0]),), q = (a = PointMass(1.0),)) => 0.5,
            (clusters = ((:out, :in) => MvNormalMeanPrecision([1.0, -1.0], [2.0 0.0; 0.0 4.0]),), q = (a = PointMass(3.0),)) => 3.0,
        ],
    )
end

# arXiv:0810.1119 solves A x = b by belief propagation, with the self-potential
# NormalWeightedMeanPrecision(b_i, A_ii) and the edge potential GaussianCoupling(x_i, x_j, -A_ij).
@testitem "rules:GaussianCoupling:solving a linear system" tags = [:rules] begin
    using GaussianCouplingMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase

    function gabp(A, b; iterations = 100)
        n = size(A, 1)
        prior = [NormalWeightedMeanPrecision(b[i], A[i, i]) for i in 1:n]
        neighbours = [[j for j in 1:n if j != i && !iszero(A[i, j])] for i in 1:n]
        # message[i, j] from variable i to variable j, uninformative to start.
        message = [NormalWeightedMeanPrecision(0.0, 0.0) for _ in 1:n, _ in 1:n]
        collect_into(i, exclude) = foldl((acc, k) -> prod(GenericProd(), acc, message[k, i]), filter(!=(exclude), neighbours[i]); init = prior[i])
        for _ in 1:iterations, i in 1:n, j in neighbours[i]
            message[i, j] = call_message_update_rule(GaussianCoupling, :in; m = (out = collect_into(i, j),), q = (a = PointMass(-A[i, j]),))
        end
        return map(i -> collect_into(i, 0), 1:n)
    end
    diagonal_of_inverse(A) = [inv(A)[i, i] for i in axes(A, 1)]

    # Acyclic: the means and the variances are exact.
    A, b = [2.0 0.5; 0.5 3.0], [1.0, 2.0]
    q = gabp(A, b)
    @test mean.(q) ≈ A \ b
    @test var.(q) ≈ diagonal_of_inverse(A)

    # With a cycle, strictly diagonally dominant so it converges: the means are exact, the
    # variances only walk-sum approximations.
    A, b = [4.0 1.0 0.5; 1.0 5.0 1.5; 0.5 1.5 6.0], [1.0, 2.0, 3.0]
    q = gabp(A, b)
    @test mean.(q) ≈ A \ b
    @test all(>(0), var.(q)) && !(var.(q) ≈ diagonal_of_inverse(A))

    A, b = [3.0 -1.0 -0.5; -1.0 4.0 -1.0; -0.5 -1.0 5.0], [-1.0, 2.0, 0.5]
    @test mean.(gabp(A, b)) ≈ A \ b
end
