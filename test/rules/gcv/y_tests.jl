
@testitem "rules:GCV:y" setup = [GCVRulesTestUtils] begin
    using ReactiveMP, BayesBase, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules, ExponentialLinearQuadratic, GaussHermiteCubature

    expected_A     = GCVRulesTestUtils.expected_A
    expected_B     = GCVRulesTestUtils.expected_B
    default_meta   = GCVRulesTestUtils.default_meta
    parameter_sets = GCVRulesTestUtils.parameter_sets

    # `y ~ N(x, exp(κz + ω))`, so the message to `y` convolves the incoming belief about `x`
    # with the effective noise. The effective noise *precision* is ⟨e^{-(κz+ω)}⟩ = A·B, hence
    # the added variance is `inv(A·B)`:
    #
    #   from a message  m_x:  N(⟨x⟩, Var(x) + 1/(A·B))
    #   from a marginal q_x:  N(⟨x⟩,          1/(A·B))
    #
    # The marginal variant omits `Var(x)` because under structured VMP the incoming `q_x`
    # marginal already accounts for the backward flow; only its mean is used.
    meta = default_meta()

    @testset "Belief-propagation-style: (m_x, q_z, q_κ, q_ω)" begin
        for params in parameter_sets()
            (; q_x, q_z, q_κ, q_ω) = params
            m_x_mean, m_x_var = mean_var(q_x)
            noise_precision = expected_A(q_ω) * expected_B(q_z, q_κ)

            msg = @call_rule GCV(:y, Marginalisation) (
                m_x = q_x, q_z = q_z, q_κ = q_κ, q_ω = q_ω, meta = meta
            )

            @test msg isa NormalMeanVariance
            @test mean(msg) ≈ m_x_mean
            @test var(msg) ≈ m_x_var + inv(noise_precision)
        end
    end

    @testset "Variational: (q_x, q_z, q_κ, q_ω)" begin
        for params in parameter_sets()
            (; q_x, q_z, q_κ, q_ω) = params
            noise_precision = expected_A(q_ω) * expected_B(q_z, q_κ)

            msg = @call_rule GCV(:y, Marginalisation) (
                q_x = q_x, q_z = q_z, q_κ = q_κ, q_ω = q_ω, meta = meta
            )

            @test msg isa NormalMeanVariance
            @test mean(msg) ≈ mean(q_x)
            @test var(msg) ≈ inv(noise_precision)
        end
    end

    @testset "Type promotion, against fully worked-out constants" begin
        # One case with every intermediate spelled out, so the reference is auditable without
        # running the helper functions, and `check_type_promotion` exercises Float32/BigFloat.
        #
        #   q_ω = N(1.2, 0.5)  → A    = exp(-1.2 + 0.5/2)                    = exp(-0.95)
        #   q_z = N(0.5, 0.7)
        #   q_κ = N(0.8, 0.4)  → ξ    = 0.8²·0.7 + 0.5²·0.4 + 0.7·0.4        = 0.828
        #                        B    = exp(-0.8·0.5 + 0.828/2)              = exp(0.014)
        #                        A·B  = exp(-0.95 + 0.014)                   = exp(-0.936)
        #   q_x = N(1.0, 2.0)  → var  = 2.0 + exp(0.936)
        @test_rules [check_type_promotion = true] GCV(
            :y, Marginalisation
        ) [(
            input = (
                m_x = NormalMeanVariance(1.0, 2.0),
                q_z = NormalMeanVariance(0.5, 0.7),
                q_κ = NormalMeanVariance(0.8, 0.4),
                q_ω = NormalMeanVariance(1.2, 0.5),
                meta = GCVMetadata(GaussHermiteCubature(20)),
            ),
            output = NormalMeanVariance(1.0, 2.0 + exp(0.936)),
        )]
    end

    @testset "The two variants differ by exactly Var(x)" begin
        # An exact structural relation between the two methods, independent of the reference
        # values above: the message-based variant adds the incoming variance, the
        # marginal-based one does not.
        for params in parameter_sets()
            (; q_x, q_z, q_κ, q_ω) = params

            from_message = @call_rule GCV(:y, Marginalisation) (
                m_x = q_x, q_z = q_z, q_κ = q_κ, q_ω = q_ω, meta = meta
            )
            from_marginal = @call_rule GCV(:y, Marginalisation) (
                q_x = q_x, q_z = q_z, q_κ = q_κ, q_ω = q_ω, meta = meta
            )

            @test mean(from_message) ≈ mean(from_marginal)
            @test var(from_message) - var(from_marginal) ≈ var(q_x)
        end
    end

    @testset "y and x rules are symmetric" begin
        # The likelihood depends on `y` and `x` only through `y - x`, so the `:y` and `:x`
        # rules must produce the same message from the same incoming belief.
        for params in parameter_sets()
            (; q_x, q_z, q_κ, q_ω) = params

            to_y = @call_rule GCV(:y, Marginalisation) (
                m_x = q_x, q_z = q_z, q_κ = q_κ, q_ω = q_ω, meta = meta
            )
            to_x = @call_rule GCV(:x, Marginalisation) (
                m_y = q_x, q_z = q_z, q_κ = q_κ, q_ω = q_ω, meta = meta
            )

            @test mean(to_y) ≈ mean(to_x)
            @test var(to_y) ≈ var(to_x)

            to_y_q = @call_rule GCV(:y, Marginalisation) (
                q_x = q_x, q_z = q_z, q_κ = q_κ, q_ω = q_ω, meta = meta
            )
            to_x_q = @call_rule GCV(:x, Marginalisation) (
                q_y = q_x, q_z = q_z, q_κ = q_κ, q_ω = q_ω, meta = meta
            )

            @test mean(to_y_q) ≈ mean(to_x_q)
            @test var(to_y_q) ≈ var(to_x_q)
        end
    end

    @testset "Accepts an ExponentialLinearQuadratic incoming message" begin
        # `m_x` is typed `UniNormalOrExpLinQuad`, so an `ExponentialLinearQuadratic` -- as
        # produced by the `:ω`/`:κ`/`:z` rules on a neighbouring edge -- must be accepted and
        # reduced through its cubature-approximated moments.
        (; q_z, q_κ, q_ω) = first(parameter_sets())
        elq = ExponentialLinearQuadratic(
            GaussHermiteCubature(20), 1.0, 1.0, -1.0, 0.0
        )
        elq_mean, elq_var = mean_var(elq)

        msg = @call_rule GCV(:y, Marginalisation) (
            m_x = elq, q_z = q_z, q_κ = q_κ, q_ω = q_ω, meta = meta
        )

        @test msg isa NormalMeanVariance
        @test mean(msg) ≈ elq_mean
        @test var(msg) ≈ elq_var + inv(expected_A(q_ω) * expected_B(q_z, q_κ))
    end
end
