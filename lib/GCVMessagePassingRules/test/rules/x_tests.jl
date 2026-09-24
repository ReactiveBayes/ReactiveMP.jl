@testitem "rules:GCV:x" tags = [:rules] setup = [GCVRulesTestUtils] begin
    using GCVMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    using MessagePassingRulesApproximations: GaussHermiteCubature
    using .GCVRulesTestUtils: expected_A, expected_B, default_algorithm, parameter_sets, test_elq

    # Mirror image of the `:y` rule -- see `y_tests.jl` for the derivation. The likelihood
    # depends on `y` and `x` only through `y - x`, so the two rules are structurally identical
    # with the roles swapped; the cross-rule symmetry itself is asserted in `y_tests.jl`.
    algorithm = default_algorithm()
    to_x(; kwargs...) = call_message_update_rule(GCV, :x; algorithm, kwargs...)

    @testset "Belief-propagation-style: (m_y, q_z, q_κ, q_ω)" begin
        for (; q_y, q_z, q_κ, q_ω) in parameter_sets()
            msg = to_x(m = (y = q_y,), q = (z = q_z, κ = q_κ, ω = q_ω))
            @test msg isa NormalMeanVariance
            @test mean(msg) ≈ mean(q_y)
            @test var(msg) ≈ var(q_y) + inv(expected_A(q_ω) * expected_B(q_z, q_κ))
        end
    end

    @testset "Variational: (q_y, q_z, q_κ, q_ω)" begin
        for (; q_y, q_z, q_κ, q_ω) in parameter_sets()
            msg = to_x(q = (y = q_y, z = q_z, κ = q_κ, ω = q_ω))
            @test msg isa NormalMeanVariance
            @test mean(msg) ≈ mean(q_y)
            @test var(msg) ≈ inv(expected_A(q_ω) * expected_B(q_z, q_κ))
        end
    end

    @testset "Type promotion, against fully worked-out constants" begin
        # Same intermediates as the `:y` type-promotion case (A·B = exp(-0.936)), with
        # q_y = N(3.0, 1.0) → var = 1.0 + exp(0.936).
        @test_message_update_rule(
            node = GCV, target = :x, algorithm = algorithm,
            cases = [
                (
                    m = (y = NormalMeanVariance(3.0, 1.0),),
                    q = (z = NormalMeanVariance(0.5, 0.7), κ = NormalMeanVariance(0.8, 0.4), ω = NormalMeanVariance(1.2, 0.5)),
                ) => NormalMeanVariance(3.0, 1.0 + exp(0.936)),
                (
                    q = (y = NormalMeanVariance(3.0, 1.0), z = NormalMeanVariance(0.5, 0.7), κ = NormalMeanVariance(0.8, 0.4), ω = NormalMeanVariance(1.2, 0.5)),
                ) => NormalMeanVariance(3.0, exp(0.936)),
            ],
        )
    end

    @testset "The two variants differ by exactly Var(y)" begin
        for (; q_y, q_z, q_κ, q_ω) in parameter_sets()
            from_message = to_x(m = (y = q_y,), q = (z = q_z, κ = q_κ, ω = q_ω))
            from_marginal = to_x(q = (y = q_y, z = q_z, κ = q_κ, ω = q_ω))
            @test mean(from_message) ≈ mean(from_marginal)
            @test var(from_message) - var(from_marginal) ≈ var(q_y)
        end
    end

    @testset "Accepts an ExponentialLinearQuadratic incoming message" begin
        (; q_z, q_κ, q_ω) = first(parameter_sets())
        elq = test_elq()
        elq_mean, elq_var = mean_var(elq)
        msg = to_x(m = (y = elq,), q = (z = q_z, κ = q_κ, ω = q_ω))
        @test msg isa NormalMeanVariance
        @test mean(msg) ≈ elq_mean
        @test var(msg) ≈ elq_var + inv(expected_A(q_ω) * expected_B(q_z, q_κ))
    end
end
