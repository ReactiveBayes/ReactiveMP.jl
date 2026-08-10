
@testitem "rules:GCV:ω" begin
    using ReactiveMP, BayesBase, ExponentialFamily, Distributions, Random

    import ReactiveMP:
        ExponentialLinearQuadratic, GCVMetadata, GaussHermiteCubature

    # The `GCV` node is `p(y | x, z, κ, ω) = N(y | x, exp(κz + ω))`, i.e. the *variance*
    # is `exp(κz + ω)`. (This is pinned independently by the node's own `@average_energy`,
    # which contributes `½[log2π + ⟨κz + ω⟩ + ⟨(y − x)²⟩·⟨e^{−κz}⟩·⟨e^{−ω}⟩]`.) Hence
    #
    #     −log p = ½[log2π + (κz + ω) + (y − x)²·e^{−(κz + ω)}]
    #
    # Marginalising `y, x, z, κ` and keeping only the `ω`-dependent terms:
    #
    #     −log f(ω) = ½[ω + ψ·⟨e^{−κz}⟩·e^{−ω}] + const,     ψ = ⟨(y − x)²⟩
    #
    # `ExponentialLinearQuadratic(a, b, c, d)` has `logpdf = −(a·ω + b·exp(c·ω + d·ω²/2))/2`,
    # so matching term by term gives the coefficients below. Note `ω` enters the linear term
    # with a *constant* coefficient of one and does not appear inside the exponential's
    # quadratic slot at all, so `a = 1`, `c = -1` and `d = 0` — none of them depend on `z`.
    #
    # This is exactly what issue #621 was about: the mean-field method had been copy-pasted
    # from the `:κ` rule and carried `a = ⟨z⟩, c = -⟨z⟩, d = Var(z)`.
    function gcv_omega_reference(psi, q_z, q_κ)
        m_z, v_z = mean_var(q_z)
        m_κ, v_κ = mean_var(q_κ)
        # Var(κz) for independent κ, z
        var_κz = m_κ^2 * v_z + m_z^2 * v_κ + v_κ * v_z
        return (1, psi * exp(-m_κ * m_z + var_κz / 2), -1, 0)
    end

    coefficients(d::ExponentialLinearQuadratic) = (d.a, d.b, d.c, d.d)

    meta = GCVMetadata(GaussHermiteCubature(20))

    @testset "Mean-field: (q_y, q_x, q_z, q_κ)" begin
        # `z` is deliberately given a mean far from 1 and a non-zero variance: those are
        # precisely the values that make the copy-pasted `:κ` coefficients differ from the
        # correct ones. With `⟨z⟩ = 1` and `Var(z) = 0` the bug would be invisible.
        for (m_y, v_y, m_x, v_x, m_z, v_z, m_κ, v_κ) in (
            (3.0, 1.0, 1.0, 2.0, 0.5, 0.7, 0.8, 0.4),
            (0.0, 1.0, 0.0, 1.0, 2.0, 0.3, 1.0, 0.0),
            (-1.5, 0.25, 2.5, 0.5, -0.75, 1.25, 0.3, 0.9),
        )
            q_y = NormalMeanVariance(m_y, v_y)
            q_x = NormalMeanVariance(m_x, v_x)
            q_z = NormalMeanVariance(m_z, v_z)
            q_κ = NormalMeanVariance(m_κ, v_κ)

            psi = (m_y - m_x)^2 + v_y + v_x

            msg = @call_rule GCV(:ω, Marginalisation) (
                q_y = q_y, q_x = q_x, q_z = q_z, q_κ = q_κ, meta = meta
            )

            @test msg isa ExponentialLinearQuadratic
            @test all(
                coefficients(msg) .≈ gcv_omega_reference(psi, q_z, q_κ)
            )

            # Pin the three `z`-independent coefficients explicitly, so a future
            # regression that reintroduces a `z`-dependence fails loudly here.
            @test msg.a == 1
            @test msg.c == -1
            @test msg.d == 0
        end
    end

    @testset "Structured: (q_y_x, q_z, q_κ)" begin
        for (m_y, m_x, V, m_z, v_z, m_κ, v_κ) in (
            (3.0, 1.0, [1.0 0.3; 0.3 2.0], 0.5, 0.7, 0.8, 0.4),
            (-1.5, 2.5, [0.25 -0.1; -0.1 0.5], -0.75, 1.25, 0.3, 0.9),
        )
            q_y_x = MvNormalMeanCovariance([m_y, m_x], V)
            q_z   = NormalMeanVariance(m_z, v_z)
            q_κ   = NormalMeanVariance(m_κ, v_κ)

            # ⟨(y − x)²⟩ under the joint posterior
            psi = (m_y - m_x)^2 + V[1, 1] + V[2, 2] - V[1, 2] - V[2, 1]

            msg = @call_rule GCV(:ω, Marginalisation) (
                q_y_x = q_y_x, q_z = q_z, q_κ = q_κ, meta = meta
            )

            @test msg isa ExponentialLinearQuadratic
            @test all(
                coefficients(msg) .≈ gcv_omega_reference(psi, q_z, q_κ)
            )
            @test msg.a == 1
            @test msg.c == -1
            @test msg.d == 0
        end
    end

    @testset "Mean-field and structured variants agree at zero y-x covariance" begin
        # With no posterior covariance between `y` and `x` the structured `q(y, x)` variant
        # reduces to the fully factorized mean-field one, so the two methods must return
        # *identical* coefficients. This is the invariant that the #621 copy-paste broke:
        # the structured method was correct while the mean-field one was not.
        for (m_y, v_y, m_x, v_x, m_z, v_z, m_κ, v_κ) in (
            (3.0, 1.0, 1.0, 2.0, 0.5, 0.7, 0.8, 0.4),
            (0.4, 0.6, 0.5, 0.3, 2.0, 0.3, 1.2, 0.25),
            (-1.5, 0.25, 2.5, 0.5, -0.75, 1.25, 0.3, 0.9),
        )
            q_y   = NormalMeanVariance(m_y, v_y)
            q_x   = NormalMeanVariance(m_x, v_x)
            q_y_x = MvNormalMeanCovariance([m_y, m_x], [v_y 0.0; 0.0 v_x])
            q_z   = NormalMeanVariance(m_z, v_z)
            q_κ   = NormalMeanVariance(m_κ, v_κ)

            meanfield = @call_rule GCV(:ω, Marginalisation) (
                q_y = q_y, q_x = q_x, q_z = q_z, q_κ = q_κ, meta = meta
            )
            structured = @call_rule GCV(:ω, Marginalisation) (
                q_y_x = q_y_x, q_z = q_z, q_κ = q_κ, meta = meta
            )

            @test all(coefficients(meanfield) .≈ coefficients(structured))
        end
    end

    @testset "ω message does not reuse the κ message's coefficients" begin
        # Direct anti-regression for #621. The `:κ` rule legitimately carries
        # `a = ⟨z⟩, c = -⟨z⟩, d = Var(z)`; the `:ω` rule must not. Asserting the
        # inequality (rather than only the correct values) documents the specific
        # failure mode and keeps failing even if the reference above were ever
        # mis-transcribed in the same direction as the bug.
        q_y = NormalMeanVariance(3.0, 1.0)
        q_x = NormalMeanVariance(1.0, 2.0)
        q_z = NormalMeanVariance(0.5, 0.7) # ⟨z⟩ ≠ 1, Var(z) ≠ 0
        q_κ = NormalMeanVariance(0.8, 0.4)
        q_ω = NormalMeanVariance(1.2, 0.5)

        ω_msg = @call_rule GCV(:ω, Marginalisation) (
            q_y = q_y, q_x = q_x, q_z = q_z, q_κ = q_κ, meta = meta
        )
        κ_msg = @call_rule GCV(:κ, Marginalisation) (
            q_y = q_y, q_x = q_x, q_z = q_z, q_ω = q_ω, meta = meta
        )

        @test (ω_msg.a, ω_msg.c, ω_msg.d) != (κ_msg.a, κ_msg.c, κ_msg.d)
        # ... and specifically that it is not carrying `z`'s moments
        @test ω_msg.a != mean(q_z)
        @test ω_msg.d != var(q_z)
    end

    @testset "Type promotion" begin
        # `a`, `c` and `d` are built from `one`/`zero` of the running float type rather
        # than hardcoded `Float64` literals, so a `Float32` input stays `Float32`.
        msg = @call_rule GCV(:ω, Marginalisation) (
            q_y = NormalMeanVariance(3.0f0, 1.0f0),
            q_x = NormalMeanVariance(1.0f0, 2.0f0),
            q_z = NormalMeanVariance(0.5f0, 0.7f0),
            q_κ = NormalMeanVariance(0.8f0, 0.4f0),
            meta = meta,
        )
        @test eltype(msg) === Float32
    end
end
