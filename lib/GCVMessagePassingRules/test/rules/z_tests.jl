@testitem "rules:GCV:z" tags = [:rules] setup = [GCVRulesTestUtils] begin
    using GCVMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    using .GCVRulesTestUtils: expected_A, expected_psi, coefficients, default_algorithm, parameter_sets

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

    algorithm = default_algorithm()
    to_z(; kwargs...) = call_message_update_rule(GCV, :z; algorithm, kwargs...)
    to_κ(; kwargs...) = call_message_update_rule(GCV, :κ; algorithm, kwargs...)

    @testset "Mean-field: (q_y, q_x, q_κ, q_ω)" begin
        for (; q_y, q_x, q_κ, q_ω) in parameter_sets()
            msg = to_z(q = (y = q_y, x = q_x, κ = q_κ, ω = q_ω))
            @test msg isa ExponentialLinearQuadratic
            @test all(coefficients(msg) .≈ reference(expected_psi(q_y, q_x), q_κ, q_ω))
            @test msg.a ≈ -msg.c
        end
    end

    @testset "Structured: (q_y_x, q_κ, q_ω)" begin
        for (m, V) in (([3.0, 1.0], [1.0 0.3; 0.3 2.0]), ([-1.5, 2.5], [0.25 -0.1; -0.1 0.5]))
            q_y_x = MvNormalMeanCovariance(m, V)
            (; q_κ, q_ω) = first(parameter_sets())
            msg = to_z(clusters = ((:y, :x) => q_y_x,), q = (κ = q_κ, ω = q_ω))
            @test msg isa ExponentialLinearQuadratic
            @test all(coefficients(msg) .≈ reference(expected_psi(q_y_x), q_κ, q_ω))
            @test msg.a ≈ -msg.c
        end
    end

    @testset "z and κ rules are mirror images under swapping their beliefs" begin
        # Because the likelihood sees only the product `κz`, feeding belief `d` as `q_κ` to the
        # `:z` rule must give the same message as feeding `d` as `q_z` to the `:κ` rule.
        for (; q_y, q_x, q_z, q_ω) in parameter_sets()
            msg_z = to_z(q = (y = q_y, x = q_x, κ = q_z, ω = q_ω))
            msg_κ = to_κ(q = (y = q_y, x = q_x, z = q_z, ω = q_ω))
            @test all(coefficients(msg_z) .≈ coefficients(msg_κ))
        end
    end

    @testset "Type promotion" begin
        msg = to_z(q = (y = NormalMeanVariance(3.0f0, 1.0f0), x = NormalMeanVariance(1.0f0, 2.0f0), κ = NormalMeanVariance(0.8f0, 0.4f0), ω = NormalMeanVariance(1.2f0, 0.5f0)))
        @test eltype(msg) === Float32
    end
end
