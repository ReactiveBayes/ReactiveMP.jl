
@testitem "rules:GCV:z" setup = [GCVRulesTestUtils] begin
    using ReactiveMP, BayesBase, ExponentialFamily, Distributions

    import ReactiveMP: ExponentialLinearQuadratic

    expected_A     = GCVRulesTestUtils.expected_A
    expected_psi   = GCVRulesTestUtils.expected_psi
    coefficients   = GCVRulesTestUtils.coefficients
    default_meta   = GCVRulesTestUtils.default_meta
    parameter_sets = GCVRulesTestUtils.parameter_sets

    # Exactly the `:κ` derivation with the roles of `κ` and `z` exchanged -- the likelihood
    # depends on them only through the product `κz`, so the two rules are mirror images:
    #
    #     -log f(z) = ½[⟨κ⟩·z + ψ·⟨e^{-ω}⟩·exp(-z⟨κ⟩ + z²Var(κ)/2)] + const
    #
    # giving `a = ⟨κ⟩`, `b = ψ·A`, `c = -⟨κ⟩`, `d = Var(κ)`.
    function reference(psi, q_κ, q_ω)
        m_κ, v_κ = mean_var(q_κ)
        return (m_κ, psi * expected_A(q_ω), -m_κ, v_κ)
    end

    meta = default_meta()

    @testset "Mean-field: (q_y, q_x, q_κ, q_ω)" begin
        for params in parameter_sets()
            (; q_y, q_x, q_κ, q_ω) = params

            msg = @call_rule GCV(:z, Marginalisation) (
                q_y = q_y, q_x = q_x, q_κ = q_κ, q_ω = q_ω, meta = meta
            )

            @test msg isa ExponentialLinearQuadratic
            @test all(
                coefficients(msg) .≈ reference(expected_psi(q_y, q_x), q_κ, q_ω)
            )
            @test msg.a ≈ -msg.c
        end
    end

    @testset "Structured: (q_y_x, q_κ, q_ω)" begin
        for (m, V) in (
            ([3.0, 1.0], [1.0 0.3; 0.3 2.0]),
            ([-1.5, 2.5], [0.25 -0.1; -0.1 0.5]),
        )
            q_y_x = MvNormalMeanCovariance(m, V)
            (; q_κ, q_ω) = first(parameter_sets())

            msg = @call_rule GCV(:z, Marginalisation) (
                q_y_x = q_y_x, q_κ = q_κ, q_ω = q_ω, meta = meta
            )

            @test msg isa ExponentialLinearQuadratic
            @test all(
                coefficients(msg) .≈ reference(expected_psi(q_y_x), q_κ, q_ω)
            )
            @test msg.a ≈ -msg.c
        end
    end

    @testset "z and κ rules are mirror images under swapping their beliefs" begin
        # Because the likelihood sees only the product `κz`, feeding belief `d` as `q_κ` to the
        # `:z` rule must give the same message as feeding `d` as `q_z` to the `:κ` rule.
        for params in parameter_sets()
            (; q_y, q_x, q_z, q_ω) = params

            to_z = @call_rule GCV(:z, Marginalisation) (
                q_y = q_y, q_x = q_x, q_κ = q_z, q_ω = q_ω, meta = meta
            )
            to_κ = @call_rule GCV(:κ, Marginalisation) (
                q_y = q_y, q_x = q_x, q_z = q_z, q_ω = q_ω, meta = meta
            )

            @test all(coefficients(to_z) .≈ coefficients(to_κ))
        end
    end

    @testset "Type promotion" begin
        msg = @call_rule GCV(:z, Marginalisation) (
            q_y = NormalMeanVariance(3.0f0, 1.0f0),
            q_x = NormalMeanVariance(1.0f0, 2.0f0),
            q_κ = NormalMeanVariance(0.8f0, 0.4f0),
            q_ω = NormalMeanVariance(1.2f0, 0.5f0),
            meta = meta,
        )
        @test eltype(msg) === Float32
    end
end
