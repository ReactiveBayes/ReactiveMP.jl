@testitem "GaussianCouplingNode" begin
    using ReactiveMP, BayesBase, ExponentialFamily
    using LinearAlgebra

    @testset "Node metadata" begin
        @test ReactiveMP.sdtype(GaussianCoupling) === Stochastic()
        @test ReactiveMP.interfaces(GaussianCoupling) === Val((:out, :in, :a))
    end

    @testset "AverageEnergy" begin
        # ⟨-log φ⟩ = -E[a] ⋅ E_{q(out, in)}[out ⋅ in] = -E[a] ⋅ (V[1, 2] + m[1] ⋅ m[2])
        energy(q_out_in, q_a) = score(
            AverageEnergy(),
            GaussianCoupling,
            Val{(:out_in, :a)}(),
            (Marginal(q_out_in, false, false), Marginal(q_a, false, false)),
            nothing,
        )

        @test energy(
            MvNormalMeanCovariance([1.0, 2.0], [2.0 0.5; 0.5 3.0]),
            PointMass(2.0),
        ) ≈ -5.0
        # Linear in `a`: flipping the sign of the coupling flips the sign of the energy.
        @test energy(
            MvNormalMeanCovariance([1.0, 2.0], [2.0 0.5; 0.5 3.0]),
            PointMass(-2.0),
        ) ≈ 5.0
        # A zero coupling contributes nothing.
        @test energy(
            MvNormalMeanCovariance([1.0, 2.0], [2.0 0.5; 0.5 3.0]),
            PointMass(0.0),
        ) ≈ 0.0
        # The covariance term matters even when the means are zero.
        @test energy(
            MvNormalMeanCovariance([0.0, 0.0], [1.0 -0.5; -0.5 1.0]),
            PointMass(1.0),
        ) ≈ 0.5
        @test energy(
            MvNormalMeanPrecision([1.0, -1.0], [2.0 0.0; 0.0 4.0]),
            PointMass(3.0),
        ) ≈ 3.0

        @testset "Cross-check against NormalMeanPrecision" begin
            # Since N(out; in, w⁻¹) ∝ exp(-w⋅out²/2) ⋅ exp(w⋅out⋅in) ⋅ exp(-w⋅in²/2),
            #   ⟨-log N⟩ = -log(w)/2 + log(2π)/2 + w⋅(E[out²] + E[in²])/2 + ⟨-log φ⟩
            # with a = w. This ties the GaussianCoupling energy to an independently tested node.
            for (q_out_in, w) in (
                (MvNormalMeanCovariance([1.0, 2.0], [2.0 0.5; 0.5 3.0]), 2.0),
                (
                    MvNormalMeanCovariance([-1.0, 0.5], [1.0 -0.25; -0.25 0.5]),
                    0.75,
                ),
            )
                m, V = mean_cov(q_out_in)
                expected = score(
                    AverageEnergy(),
                    NormalMeanPrecision,
                    Val{(:out_μ, :τ)}(),
                    (
                        Marginal(q_out_in, false, false),
                        Marginal(PointMass(w), false, false),
                    ),
                    nothing,
                )
                second_moments = (V[1, 1] + abs2(m[1])) + (V[2, 2] + abs2(m[2]))
                @test -log(w) / 2 +
                      log(2π) / 2 +
                      w * second_moments / 2 +
                      energy(q_out_in, PointMass(w)) ≈ expected
            end
        end
    end

    @testset "GaBP: solving a linear system" begin
        # arXiv:0810.1119 casts `A x = b` as message passing with
        #   self-potential  exp(b_i⋅x_i - A_ii⋅x_i²/2) = NormalWeightedMeanPrecision(b_i, A_ii)
        #   edge potential  exp(-x_i⋅A_ij⋅x_j)         = GaussianCoupling(x_i, x_j, -A_ij)
        function gabp(A, b; iterations = 100)
            n = size(A, 1)
            prior = [NormalWeightedMeanPrecision(b[i], A[i, i]) for i in 1:n]
            nbrs = [[j for j in 1:n if j != i && !iszero(A[i, j])] for i in 1:n]
            # msg[i, j] — message from variable `i` to variable `j`; a zero-precision
            # Gaussian is the uninformative initial message.
            msg = [NormalWeightedMeanPrecision(0.0, 0.0) for _ in 1:n, _ in 1:n]
            collect_into(i, exclude) = foldl(
                (acc, k) -> prod(GenericProd(), acc, msg[k, i]),
                filter(!=(exclude), nbrs[i]);
                init = prior[i],
            )
            for _ in 1:iterations, i in 1:n, j in nbrs[i]
                msg[i, j] = @call_rule GaussianCoupling(:in, Marginalisation) (
                    m_out = collect_into(i, j), q_a = PointMass(-A[i, j])
                )
            end
            return map(i -> collect_into(i, 0), 1:n)
        end

        @testset "Acyclic graph: means and variances are both exact" begin
            A = [2.0 0.5; 0.5 3.0]
            b = [1.0, 2.0]
            q = gabp(A, b)
            @test mean.(q) ≈ A \ b
            @test var.(q) ≈ diag(inv(A))
        end

        @testset "Loopy graph: means are exact, variances are not" begin
            # Strictly diagonally dominant, so GaBP converges (Theorem 12).
            A = [4.0 1.0 0.5; 1.0 5.0 1.5; 0.5 1.5 6.0]
            b = [1.0, 2.0, 3.0]
            @test all(i -> A[i, i] > sum(abs, A[i, :]) - A[i, i], 1:3)

            q = gabp(A, b)
            @test mean.(q) ≈ A \ b
            # On a graph with cycles the variances are only walk-sum approximations of
            # diag(inv(A)) — they stay positive and finite, but they are not exact.
            @test all(>(0), var.(q))
            @test !(var.(q) ≈ diag(inv(A)))
        end

        @testset "Negative off-diagonals" begin
            A = [3.0 -1.0 -0.5; -1.0 4.0 -1.0; -0.5 -1.0 5.0]
            b = [-1.0, 2.0, 0.5]
            @test mean.(gabp(A, b)) ≈ A \ b
        end
    end
end
