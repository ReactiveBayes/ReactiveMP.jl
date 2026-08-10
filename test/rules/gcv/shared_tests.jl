
@testitem "rules:GCV: effective noise moments" setup = [GCVRulesTestUtils] begin
    using ReactiveMP, BayesBase, ExponentialFamily, Distributions, StableRNGs

    expected_A     = GCVRulesTestUtils.expected_A
    expected_B     = GCVRulesTestUtils.expected_B
    parameter_sets = GCVRulesTestUtils.parameter_sets

    # These two quantities appear in every GCV rule, so it is worth establishing what they
    # are before testing the rules that consume them.

    @testset "A = ⟨e^{-ω}⟩ is exact, and matches Monte Carlo" begin
        rng = StableRNG(1234)
        for params in parameter_sets()
            q_ω = params.q_ω
            m, v = mean_var(q_ω)

            # Closed form: the mean of a lognormal.
            @test expected_A(q_ω) ≈ exp(-m + v / 2)

            # Independent Monte Carlo confirmation that the closed form is the *correct*
            # expectation and not merely the one the implementation happens to use.
            # Tolerance is the sampling noise floor for 10^6 draws, not a fudge factor.
            samples = exp.(-rand(rng, Normal(m, sqrt(v)), 10^6))
            @test mean(samples) ≈ expected_A(q_ω) rtol = 5e-3
        end
    end

    @testset "B = ⟨e^{-κz}⟩ is a documented Gaussian-moment approximation" begin
        # `κz` is a product of independent Gaussians, so it is not Gaussian and the
        # lognormal formula does not give its exact exponential moment. The node applies it
        # anyway, using Var(κz) for the variance. This test documents that gap rather than
        # asserting the approximation is exact -- if a future change replaced `B` with a
        # genuinely exact computation, this test would flag it as an intentional decision to
        # revisit rather than silently passing.
        rng = StableRNG(5678)
        params = first(parameter_sets())
        q_z, q_κ = params.q_z, params.q_κ

        m_z, v_z = mean_var(q_z)
        m_κ, v_κ = mean_var(q_κ)

        z_samples = rand(rng, Normal(m_z, sqrt(v_z)), 10^6)
        κ_samples = rand(rng, Normal(m_κ, sqrt(v_κ)), 10^6)
        truth = mean(exp.(-κ_samples .* z_samples))

        # The approximation reproduces the true variance of the exponent ...
        @test v_κ * v_z + m_κ^2 * v_z + m_z^2 * v_κ ≈
            var(κ_samples .* z_samples) rtol = 1e-2

        # ... and lands in the right ballpark, but is not exact.
        @test expected_B(q_z, q_κ) ≈ truth rtol = 0.2
        @test !isapprox(expected_B(q_z, q_κ), truth; rtol = 1e-4)
    end
end

@testitem "rules:GCV: mean-field and structured variants agree at zero y-x covariance" setup = [
    GCVRulesTestUtils
] begin
    using ReactiveMP, BayesBase, ExponentialFamily, Distributions

    coefficients         = GCVRulesTestUtils.coefficients
    block_diagonal_joint = GCVRulesTestUtils.block_diagonal_joint
    default_meta         = GCVRulesTestUtils.default_meta
    parameter_sets       = GCVRulesTestUtils.parameter_sets

    # With no posterior covariance between `y` and `x`, a structured `q(y, x)` carries exactly
    # the information of the factorized pair, so every structured rule must reduce to its
    # mean-field sibling. This needs no reference values at all -- only that the two
    # implementations agree -- which makes it the cheapest possible guard against the class of
    # copy-paste defect found in #621.
    #
    # The `:ω` case of this invariant lives in `w_tests.jl`, alongside the fix for #621 that it
    # requires; `:κ` and `:z` are covered here.
    meta = default_meta()

    for params in parameter_sets()
        (; q_y, q_x, q_z, q_κ, q_ω) = params
        q_y_x = block_diagonal_joint(q_y, q_x)

        @testset "κ" begin
            @test all(
                coefficients(
                    @call_rule GCV(:κ, Marginalisation) (
                        q_y = q_y, q_x = q_x, q_z = q_z, q_ω = q_ω, meta = meta
                    )
                ) .≈ coefficients(
                    @call_rule GCV(:κ, Marginalisation) (
                        q_y_x = q_y_x, q_z = q_z, q_ω = q_ω, meta = meta
                    )
                ),
            )
        end

        @testset "z" begin
            @test all(
                coefficients(
                    @call_rule GCV(:z, Marginalisation) (
                        q_y = q_y, q_x = q_x, q_κ = q_κ, q_ω = q_ω, meta = meta
                    )
                ) .≈ coefficients(
                    @call_rule GCV(:z, Marginalisation) (
                        q_y_x = q_y_x, q_κ = q_κ, q_ω = q_ω, meta = meta
                    )
                ),
            )
        end
    end
end

@testitem "rules:GCV: the three ExponentialLinearQuadratic rules are mutually distinct" setup = [
    GCVRulesTestUtils
] begin
    using ReactiveMP, BayesBase, ExponentialFamily, Distributions

    default_meta = GCVRulesTestUtils.default_meta

    # `:ω`, `:κ` and `:z` all return an `ExponentialLinearQuadratic` and their bodies look
    # near-identical, which is exactly how the #621 copy-paste survived review. Their
    # coefficients encode genuinely different structure:
    #
    #   :ω → (1,    ψ·B, -1,    0     )   ω multiplies nothing; constant coefficient
    #   :κ → (⟨z⟩,  ψ·A, -⟨z⟩,  Var(z))   κ multiplies z
    #   :z → (⟨κ⟩,  ψ·A, -⟨κ⟩,  Var(κ))   z multiplies κ
    #
    # so with `⟨z⟩ ≠ ⟨κ⟩` and `Var(z) ≠ Var(κ)` no two of them may coincide. The `:ω` half of
    # this check lives in `w_tests.jl` with the #621 fix it requires; `:κ` vs `:z` is here.
    meta = default_meta()

    q_y = NormalMeanVariance(3.0, 1.0)
    q_x = NormalMeanVariance(1.0, 2.0)
    q_z = NormalMeanVariance(0.5, 0.7)
    q_κ = NormalMeanVariance(0.8, 0.4)
    q_ω = NormalMeanVariance(1.2, 0.5)

    κ_msg = @call_rule GCV(:κ, Marginalisation) (
        q_y = q_y, q_x = q_x, q_z = q_z, q_ω = q_ω, meta = meta
    )
    z_msg = @call_rule GCV(:z, Marginalisation) (
        q_y = q_y, q_x = q_x, q_κ = q_κ, q_ω = q_ω, meta = meta
    )

    shape(d) = (d.a, d.c, d.d)

    @test shape(κ_msg) != shape(z_msg)

    # And each carries the moments of the variable it is *supposed* to have marginalised --
    # `:κ` integrates out `z`, `:z` integrates out `κ`, so a swap between them is caught here.
    @test (κ_msg.a, κ_msg.d) == (mean(q_z), var(q_z))
    @test (z_msg.a, z_msg.d) == (mean(q_κ), var(q_κ))

    # Both share the same `b = ψ·⟨e^{-ω}⟩`, since both marginalise `ω` the same way.
    @test κ_msg.b ≈ z_msg.b
end
