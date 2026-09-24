@testitem "rules:GCV:y" tags = [:rules] setup = [GCVRulesTestUtils] begin
    using GCVMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    using MessagePassingRulesApproximations: GaussHermiteCubature
    using .GCVRulesTestUtils: expected_A, expected_B, default_algorithm, parameter_sets, test_elq

    # `y ~ N(x, exp(κz + ω))`, so the message to `y` convolves the incoming belief about `x`
    # with the effective noise. The effective noise *precision* is ⟨e^{-(κz+ω)}⟩ = A·B, hence
    # the added variance is `inv(A·B)`:
    #
    #   from a message  m_x:  N(⟨x⟩, Var(x) + 1/(A·B))
    #   from a marginal q_x:  N(⟨x⟩,          1/(A·B))
    #
    # The marginal variant omits `Var(x)` because under structured VMP the incoming `q_x`
    # marginal already accounts for the backward flow; only its mean is used.
    algorithm = default_algorithm()
    to_y(; kwargs...) = call_message_update_rule(GCV, :y; algorithm, kwargs...)
    to_x(; kwargs...) = call_message_update_rule(GCV, :x; algorithm, kwargs...)

    @testset "Belief-propagation-style: (m_x, q_z, q_κ, q_ω)" begin
        for (; q_x, q_z, q_κ, q_ω) in parameter_sets()
            msg = to_y(m = (x = q_x,), q = (z = q_z, κ = q_κ, ω = q_ω))
            @test msg isa NormalMeanVariance
            @test mean(msg) ≈ mean(q_x)
            @test var(msg) ≈ var(q_x) + inv(expected_A(q_ω) * expected_B(q_z, q_κ))
        end
    end

    @testset "Variational: (q_x, q_z, q_κ, q_ω)" begin
        for (; q_x, q_z, q_κ, q_ω) in parameter_sets()
            msg = to_y(q = (x = q_x, z = q_z, κ = q_κ, ω = q_ω))
            @test msg isa NormalMeanVariance
            @test mean(msg) ≈ mean(q_x)
            @test var(msg) ≈ inv(expected_A(q_ω) * expected_B(q_z, q_κ))
        end
    end

    @testset "Type promotion, against fully worked-out constants" begin
        # One case with every intermediate spelled out, so the reference is auditable without
        # running the helper functions, and the table checks Float32/BigFloat.
        #
        #   q_ω = N(1.2, 0.5)  → A    = exp(-1.2 + 0.5/2)                    = exp(-0.95)
        #   q_z = N(0.5, 0.7)
        #   q_κ = N(0.8, 0.4)  → ξ    = 0.8²·0.7 + 0.5²·0.4 + 0.7·0.4        = 0.828
        #                        B    = exp(-0.8·0.5 + 0.828/2)              = exp(0.014)
        #                        A·B  = exp(-0.95 + 0.014)                   = exp(-0.936)
        #   q_x = N(1.0, 2.0)  → var  = 2.0 + exp(0.936)
        @test_message_update_rule(
            node = GCV, target = :y, algorithm = algorithm,
            cases = [
                (
                    m = (x = NormalMeanVariance(1.0, 2.0),),
                    q = (z = NormalMeanVariance(0.5, 0.7), κ = NormalMeanVariance(0.8, 0.4), ω = NormalMeanVariance(1.2, 0.5)),
                ) => NormalMeanVariance(1.0, 2.0 + exp(0.936)),
                # Under mean-field only the mean of `q(x)` enters.
                (
                    q = (x = NormalMeanVariance(1.0, 2.0), z = NormalMeanVariance(0.5, 0.7), κ = NormalMeanVariance(0.8, 0.4), ω = NormalMeanVariance(1.2, 0.5)),
                ) => NormalMeanVariance(1.0, exp(0.936)),
            ],
        )
    end

    @testset "The two variants differ by exactly Var(x)" begin
        # An exact structural relation between the two methods, independent of the reference
        # values above: the message-based variant adds the incoming variance, the
        # marginal-based one does not.
        for (; q_x, q_z, q_κ, q_ω) in parameter_sets()
            from_message = to_y(m = (x = q_x,), q = (z = q_z, κ = q_κ, ω = q_ω))
            from_marginal = to_y(q = (x = q_x, z = q_z, κ = q_κ, ω = q_ω))
            @test mean(from_message) ≈ mean(from_marginal)
            @test var(from_message) - var(from_marginal) ≈ var(q_x)
        end
    end

    @testset "y and x rules are symmetric" begin
        # The likelihood depends on `y` and `x` only through `y - x`, so the `:y` and `:x`
        # rules must produce the same message from the same incoming belief.
        for (; q_x, q_z, q_κ, q_ω) in parameter_sets()
            to_y_m = to_y(m = (x = q_x,), q = (z = q_z, κ = q_κ, ω = q_ω))
            to_x_m = to_x(m = (y = q_x,), q = (z = q_z, κ = q_κ, ω = q_ω))
            @test mean(to_y_m) ≈ mean(to_x_m)
            @test var(to_y_m) ≈ var(to_x_m)
            to_y_q = to_y(q = (x = q_x, z = q_z, κ = q_κ, ω = q_ω))
            to_x_q = to_x(q = (y = q_x, z = q_z, κ = q_κ, ω = q_ω))
            @test mean(to_y_q) ≈ mean(to_x_q)
            @test var(to_y_q) ≈ var(to_x_q)
        end
    end

    @testset "Accepts an ExponentialLinearQuadratic incoming message" begin
        # `m_x` may be an `ExponentialLinearQuadratic` -- as produced by the `:ω`/`:κ`/`:z`
        # rules on a neighbouring edge -- and is reduced through its cubature-approximated
        # moments.
        (; q_z, q_κ, q_ω) = first(parameter_sets())
        elq = test_elq()
        elq_mean, elq_var = mean_var(elq)
        msg = to_y(m = (x = elq,), q = (z = q_z, κ = q_κ, ω = q_ω))
        @test msg isa NormalMeanVariance
        @test mean(msg) ≈ elq_mean
        @test var(msg) ≈ elq_var + inv(expected_A(q_ω) * expected_B(q_z, q_κ))
    end
end
