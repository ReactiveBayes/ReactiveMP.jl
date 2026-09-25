@testitem "helpers:split_y_x" tags = [:rules] begin
    using SoftDotMessagePassingRules, ExponentialFamily, BayesBase, StableRNGs
    using SoftDotMessagePassingRules: split_y_x

    rng = StableRNG(42)
    for order in (1, 2, 3)
        A = randn(rng, order + 1, order + 1)
        V = A * A' + (order + 1) * one(A)
        m = randn(rng, order + 1)
        my, Vy, mx, Vx, Vxy = split_y_x(MvNormalMeanCovariance(m, V))
        @test my == m[1] && Vy == V[1, 1]
        if order == 1
            # Scalars for order 1.
            @test (mx, Vx, Vxy) == (m[2], V[2, 2], V[2, 1])
        else
            @test mx == m[2:end] && Vx == V[2:end, 2:end] && Vxy == V[2:end, 1]
        end
    end
end

@testitem "helpers:y_from_x" tags = [:rules] begin
    using SoftDotMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions, StableRNGs
    using SoftDotMessagePassingRules: y_from_x
    using LinearAlgebra: I

    # The reference: AR's `y` message, formed with the dense companion matrix of ⟨θ⟩
    # and the transition noise 1/⟨γ⟩ on the first component, and then its first component.
    function ar_y_first_component(m_x, q_θ, q_γ)
        mθ, Vθ = mean_cov(q_θ)
        mx, Wx = mean_invcov(m_x)
        mγ = mean(q_γ)
        order = length(mθ)
        A = [mθ'; Matrix(1.0I, order - 1, order - 1) zeros(order - 1)]
        noise = zeros(order, order)
        noise[1, 1] = inv(mγ)
        C = A * inv(Wx + mγ * Vθ)
        return (C * Wx * mx)[1], (C * A' + noise)[1, 1]
    end

    rng = StableRNG(1234)
    for order in (2, 3, 5), _ in 1:5
        B = randn(rng, order, order)
        m_x = MvNormalMeanPrecision(randn(rng, order), B * B' + order * I)
        B = randn(rng, order, order)
        q_θ = MvNormalMeanCovariance(randn(rng, order), B * B' + order * I)
        q_γ = GammaShapeRate(1.0 + rand(rng), 1.0 + rand(rng))
        my, Vy = ar_y_first_component(m_x, q_θ, q_γ)
        message = y_from_x(m_x, q_θ, q_γ)
        @test message isa NormalMeanVariance
        @test mean(message) ≈ my
        @test var(message) ≈ Vy
        # And the rule towards `y` under q(y, x) is this formula.
        @test getresult(call_message_update_rule(SoftDot, :y; m = (x = m_x,), q = (θ = q_θ, γ = q_γ))) ≈ message
    end

    # Order 1, where the companion matrix is ⟨θ⟩ itself: N(θ W_x m_x / D, θ² / D + 1/⟨γ⟩).
    m_x, q_θ, q_γ = NormalMeanPrecision(2.0, 3.0), NormalMeanVariance(0.5, 0.25), GammaShapeRate(2.0, 1.0)
    D = 3.0 + 2.0 * 0.25
    message = y_from_x(m_x, q_θ, q_γ)
    @test mean(message) ≈ 0.5 * 3.0 * 2.0 / D
    @test var(message) ≈ 0.5^2 / D + 1 / 2.0
end
