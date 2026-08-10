
@testitem "rules:GCV:κ" setup = [GCVRulesTestUtils] begin
    using ReactiveMP, BayesBase, ExponentialFamily, Distributions

    import ReactiveMP: ExponentialLinearQuadratic

    expected_A     = GCVRulesTestUtils.expected_A
    expected_psi   = GCVRulesTestUtils.expected_psi
    coefficients   = GCVRulesTestUtils.coefficients
    default_meta   = GCVRulesTestUtils.default_meta
    parameter_sets = GCVRulesTestUtils.parameter_sets

    # Keeping only the `κ`-dependent terms of `-log p = ½[log2π + (κz + ω) + ψ·e^{-(κz+ω)}]`
    # and marginalising `y, x, z, ω`:
    #
    #     -log f(κ) = ½[⟨z⟩·κ + ψ·⟨e^{-ω}⟩·⟨e^{-κz}⟩_z] + const
    #
    # where the inner expectation is over `z` only, with `κ` still free:
    #
    #     ⟨e^{-κz}⟩_z = exp(-κ⟨z⟩ + κ²Var(z)/2)     (exact: z is Gaussian, κ is a constant here)
    #
    # Matching `logpdf = -(a·κ + b·exp(c·κ + d·κ²/2))/2` term by term:
    #
    #     a = ⟨z⟩,   b = ψ·A,   c = -⟨z⟩,   d = Var(z)
    #
    # Note this expectation *is* exact, unlike the `B` used by the `:y`/`:x`/`:ω` rules --
    # here only `z` is integrated out and `κ` remains a free variable of the message.
    function reference(psi, q_z, q_ω)
        m_z, v_z = mean_var(q_z)
        return (m_z, psi * expected_A(q_ω), -m_z, v_z)
    end

    meta = default_meta()

    @testset "Mean-field: (q_y, q_x, q_z, q_ω)" begin
        for params in parameter_sets()
            (; q_y, q_x, q_z, q_ω) = params

            msg = @call_rule GCV(:κ, Marginalisation) (
                q_y = q_y, q_x = q_x, q_z = q_z, q_ω = q_ω, meta = meta
            )

            @test msg isa ExponentialLinearQuadratic
            @test all(
                coefficients(msg) .≈
                reference(expected_psi(q_y, q_x), q_z, q_ω)
            )
            # `a` and `c` are equal and opposite by construction.
            @test msg.a ≈ -msg.c
        end
    end

    @testset "Structured: (q_y_x, q_z, q_ω)" begin
        for (m, V) in (
            ([3.0, 1.0], [1.0 0.3; 0.3 2.0]),
            ([-1.5, 2.5], [0.25 -0.1; -0.1 0.5]),
        )
            q_y_x = MvNormalMeanCovariance(m, V)
            (; q_z, q_ω) = first(parameter_sets())

            msg = @call_rule GCV(:κ, Marginalisation) (
                q_y_x = q_y_x, q_z = q_z, q_ω = q_ω, meta = meta
            )

            @test msg isa ExponentialLinearQuadratic
            @test all(
                coefficients(msg) .≈ reference(expected_psi(q_y_x), q_z, q_ω)
            )
            @test msg.a ≈ -msg.c
        end
    end

    @testset "The y-x covariance enters only through ψ" begin
        # `ψ = ⟨(y-x)²⟩ = ⟨y-x⟩² + Var(y) + Var(x) - 2Cov(y,x)`, so a positive covariance
        # *reduces* ψ and hence `b`, while leaving `a`, `c`, `d` untouched. This pins that the
        # structured method routes the covariance to the right place.
        (; q_z, q_ω) = first(parameter_sets())
        m = [3.0, 1.0]

        uncorrelated = @call_rule GCV(:κ, Marginalisation) (
            q_y_x = MvNormalMeanCovariance(m, [1.0 0.0; 0.0 2.0]),
            q_z = q_z,
            q_ω = q_ω,
            meta = meta,
        )
        correlated = @call_rule GCV(:κ, Marginalisation) (
            q_y_x = MvNormalMeanCovariance(m, [1.0 0.5; 0.5 2.0]),
            q_z = q_z,
            q_ω = q_ω,
            meta = meta,
        )

        @test correlated.b < uncorrelated.b
        @test (correlated.a, correlated.c, correlated.d) ==
            (uncorrelated.a, uncorrelated.c, uncorrelated.d)
        # ψ drops by exactly 2·Cov(y, x) = 1.0, and `b = ψ·A`
        @test uncorrelated.b - correlated.b ≈ 1.0 * expected_A(q_ω)
    end

    @testset "Type promotion" begin
        msg = @call_rule GCV(:κ, Marginalisation) (
            q_y = NormalMeanVariance(3.0f0, 1.0f0),
            q_x = NormalMeanVariance(1.0f0, 2.0f0),
            q_z = NormalMeanVariance(0.5f0, 0.7f0),
            q_ω = NormalMeanVariance(1.2f0, 0.5f0),
            meta = meta,
        )
        @test eltype(msg) === Float32
    end
end
