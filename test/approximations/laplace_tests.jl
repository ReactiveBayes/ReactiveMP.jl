
@testitem "LaplaceApproximation: recovers the exact Gaussian product" begin
    using ReactiveMP, BayesBase, Distributions, ExponentialFamily, LinearAlgebra

    import ReactiveMP: LaplaceApproximation, approximate_meancov

    # For a Gaussian `g` and a Gaussian proposal the Laplace approximation is *exact*: the
    # log-density is a quadratic, so its second-order expansion at the mode reproduces it
    # exactly. That makes this the sharpest possible test of the method -- no Monte Carlo or
    # quadrature error to hide behind, and the expected values are closed-form.
    #
    # For `g(z) = exp(-(z-μ)ᵀΛ_g(z-μ)/2)` and proposal `N(m₀, Σ₀)`:
    #
    #     Λ = Λ₀ + Λ_g,    m = Λ⁻¹(Λ₀m₀ + Λ_g μ),    Σ = Λ⁻¹
    #
    # This is also the regression test for the covariance sign: before the fix the returned
    # covariance was the *negation* of the correct one, for every input -- a broken result that
    # went unnoticed because `LaplaceApproximation` had no tests at all.

    @testset "univariate (1-element vector)" begin
        # proposal N(0, 1), kernel centred at 1 with precision 1
        # ⇒ Λ = 2, Σ = 0.5, m = 0.5·1 = 0.5
        m, c = approximate_meancov(
            LaplaceApproximation(),
            (z) -> exp(-sum(abs2, z .- 1.0) / 2),
            MvNormalMeanCovariance([0.0], [1.0;;]),
        )

        @test m ≈ [0.5] atol = 1e-6
        @test c ≈ [0.5;;] atol = 1e-8
        # The covariance must be a valid one. This is the assertion that fails on `main`.
        @test isposdef(c)
        @test all(>(0), diag(c))
    end

    @testset "multivariate with correlation" begin
        Λ₀ = [2.0 0.3; 0.3 1.5]      # proposal precision
        Λg = [1.0 -0.2; -0.2 3.0]    # kernel precision
        m₀ = [0.5, -1.0]
        μg = [2.0, 1.0]

        Λ = Λ₀ + Λg
        Σ_exact = inv(Λ)
        m_exact = Σ_exact * (Λ₀ * m₀ + Λg * μg)

        m, c = approximate_meancov(
            LaplaceApproximation(),
            (z) -> exp(-dot(z - μg, Λg, z - μg) / 2),
            MvNormalMeanCovariance(m₀, inv(Λ₀)),
        )

        @test m ≈ m_exact atol = 1e-6
        @test c ≈ Σ_exact atol = 1e-8
        @test isposdef(c)
        @test issymmetric(Matrix(c)) || c ≈ transpose(c)
    end

    @testset "the covariance is positive-definite across a range of curvatures" begin
        # Sweeps the kernel precision over four orders of magnitude. Every returned covariance
        # must be positive-definite; on `main` every one of them is negative-definite.
        for λ in (1e-2, 1.0, 1e1, 1e2)
            m, c = approximate_meancov(
                LaplaceApproximation(),
                (z) -> exp(-λ * sum(abs2, z .- 0.5) / 2),
                MvNormalMeanCovariance([0.0], [1.0;;]),
            )

            @test isposdef(c)
            # Exact: Λ = 1 + λ
            @test c ≈ [inv(1 + λ);;] atol = 1e-8
        end
    end
end

@testitem "LaplaceApproximation: rejects a stationary point that is not a maximum" begin
    using ReactiveMP, BayesBase, Distributions, ExponentialFamily, LinearAlgebra

    import ReactiveMP: LaplaceApproximation, approximate_meancov

    # `Optim.converged` only reports that the optimizer stopped moving; it says nothing about
    # whether the stationary point is a strict local maximum. When it is not, `-H` is not
    # positive-definite and there is no Gaussian to fit -- previously this silently produced an
    # indefinite "covariance" (issue #635).
    #
    # A kernel that is log-*convex* along one direction inverts the curvature the proposal
    # supplies, so the total log-density has no maximum there.
    @testset "log-convex kernel is rejected with an explanatory error" begin
        # log g(z) = +2·z², so log f = 2z² - z²/2 = +1.5z²: curvature is positive, i.e. the
        # stationary point at 0 is a minimum of log f, not a maximum.
        err = try
            approximate_meancov(
                LaplaceApproximation(),
                (z) -> exp(2 * sum(abs2, z)),
                MvNormalMeanCovariance([0.0], [1.0;;]),
            )
            nothing
        catch e
            e
        end

        @test err !== nothing
        if err isa ErrorException
            # Either the explicit concavity guard fires, or the optimizer itself fails to
            # converge on an unbounded objective. Both are acceptable refusals; what matters is
            # that no invalid covariance is returned. When it is the guard, it must explain
            # itself and suggest a way forward.
            @test occursin("not locally concave", err.msg) ||
                occursin("convergence failed", err.msg)
            if occursin("not locally concave", err.msg)
                @test occursin("positive-definite", err.msg)
                @test occursin("GaussHermiteCubature", err.msg)
            end
        end
    end

    @testset "a valid case is not caught by the guard" begin
        # The guard must not reject legitimate log-concave problems.
        m, c = approximate_meancov(
            LaplaceApproximation(),
            (z) -> exp(-sum(abs2, z .- 1.0) / 2),
            MvNormalMeanCovariance([0.0], [1.0;;]),
        )
        @test isposdef(c)
    end
end
