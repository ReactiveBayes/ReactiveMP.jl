
@testitem "rules:GCV:x" setup = [GCVRulesTestUtils] begin
    using ReactiveMP, BayesBase, ExponentialFamily, Distributions

    import ReactiveMP:
        @test_rules, ExponentialLinearQuadratic, GaussHermiteCubature

    expected_A     = GCVRulesTestUtils.expected_A
    expected_B     = GCVRulesTestUtils.expected_B
    default_meta   = GCVRulesTestUtils.default_meta
    parameter_sets = GCVRulesTestUtils.parameter_sets

    # Mirror image of the `:y` rule -- see `y_tests.jl` for the derivation. The likelihood
    # depends on `y` and `x` only through `y - x`, so the two rules are structurally identical
    # with the roles swapped; the cross-rule symmetry itself is asserted in `y_tests.jl`.
    meta = default_meta()

    @testset "Belief-propagation-style: (m_y, q_z, q_κ, q_ω)" begin
        for params in parameter_sets()
            (; q_y, q_z, q_κ, q_ω) = params
            m_y_mean, m_y_var = mean_var(q_y)
            noise_precision = expected_A(q_ω) * expected_B(q_z, q_κ)

            msg = @call_rule GCV(:x, Marginalisation) (
                m_y = q_y, q_z = q_z, q_κ = q_κ, q_ω = q_ω, meta = meta
            )

            @test msg isa NormalMeanVariance
            @test mean(msg) ≈ m_y_mean
            @test var(msg) ≈ m_y_var + inv(noise_precision)
        end
    end

    @testset "Variational: (q_y, q_z, q_κ, q_ω)" begin
        for params in parameter_sets()
            (; q_y, q_z, q_κ, q_ω) = params
            noise_precision = expected_A(q_ω) * expected_B(q_z, q_κ)

            msg = @call_rule GCV(:x, Marginalisation) (
                q_y = q_y, q_z = q_z, q_κ = q_κ, q_ω = q_ω, meta = meta
            )

            @test msg isa NormalMeanVariance
            @test mean(msg) ≈ mean(q_y)
            @test var(msg) ≈ inv(noise_precision)
        end
    end

    @testset "Type promotion, against fully worked-out constants" begin
        # Same intermediates as the `:y` type-promotion case (A·B = exp(-0.936)), with
        # q_y = N(3.0, 1.0) → var = 1.0 + exp(0.936).
        @test_rules [check_type_promotion = true] GCV(:x, Marginalisation) [(
            input = (
                m_y = NormalMeanVariance(3.0, 1.0),
                q_z = NormalMeanVariance(0.5, 0.7),
                q_κ = NormalMeanVariance(0.8, 0.4),
                q_ω = NormalMeanVariance(1.2, 0.5),
                meta = GCVMetadata(GaussHermiteCubature(20)),
            ),
            output = NormalMeanVariance(3.0, 1.0 + exp(0.936)),
        )]
    end

    @testset "The two variants differ by exactly Var(y)" begin
        for params in parameter_sets()
            (; q_y, q_z, q_κ, q_ω) = params

            from_message = @call_rule GCV(:x, Marginalisation) (
                m_y = q_y, q_z = q_z, q_κ = q_κ, q_ω = q_ω, meta = meta
            )
            from_marginal = @call_rule GCV(:x, Marginalisation) (
                q_y = q_y, q_z = q_z, q_κ = q_κ, q_ω = q_ω, meta = meta
            )

            @test mean(from_message) ≈ mean(from_marginal)
            @test var(from_message) - var(from_marginal) ≈ var(q_y)
        end
    end

    @testset "Accepts an ExponentialLinearQuadratic incoming message" begin
        (; q_z, q_κ, q_ω) = first(parameter_sets())
        elq = ExponentialLinearQuadratic(
            GaussHermiteCubature(20), 1.0, 1.0, -1.0, 0.0
        )
        elq_mean, elq_var = mean_var(elq)

        msg = @call_rule GCV(:x, Marginalisation) (
            m_y = elq, q_z = q_z, q_κ = q_κ, q_ω = q_ω, meta = meta
        )

        @test msg isa NormalMeanVariance
        @test mean(msg) ≈ elq_mean
        @test var(msg) ≈ elq_var + inv(expected_A(q_ω) * expected_B(q_z, q_κ))
    end
end
