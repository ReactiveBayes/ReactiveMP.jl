
@testitem "rules:ConjugateAR:y (delegates to AR)" begin
    using ReactiveMP, BayesBase, ExponentialFamily, Distributions, LinearAlgebra, StableRNGs

    import ReactiveMP: @call_rule, conjugatear_effective_marginals

    same_normal(a, b; atol = 1e-8) =
        isapprox(mean(a), mean(b); atol = atol) && isapprox(cov(a), cov(b); atol = atol)

    @testset "y message equals AR(:y) with effective (q_θ, q_γ)" begin
        rng = StableRNG(33)
        for order in (1, 2)
            meta = ARMeta(Multivariate, order, ARsafe())
            B = randn(rng, order, order)
            q_w = MvNormalGamma(randn(rng, order), B * B' + diageye(order), 2.0 + rand(rng), 1.0 + rand(rng))
            q_θ, q_γ = conjugatear_effective_marginals(q_w)
            m_x = MvNormalMeanCovariance(randn(rng, order), diageye(order))

            got = @call_rule ConjugateAR(:y, Marginalisation) (m_x = m_x, q_w = q_w, meta = meta)
            exp = @call_rule AR(:y, Marginalisation) (m_x = m_x, q_θ = q_θ, q_γ = q_γ, meta = meta)
            @test same_normal(got, exp)
        end
    end

    @testset "y message (q_x variant) equals AR(:y)" begin
        rng = StableRNG(34)
        for order in (1, 2)
            meta = ARMeta(Multivariate, order, ARsafe())
            B = randn(rng, order, order)
            q_w = MvNormalGamma(randn(rng, order), B * B' + diageye(order), 2.0 + rand(rng), 1.0 + rand(rng))
            q_θ, q_γ = conjugatear_effective_marginals(q_w)
            q_x = MvNormalMeanCovariance(randn(rng, order), diageye(order))

            got = @call_rule ConjugateAR(:y, Marginalisation) (q_x = q_x, q_w = q_w, meta = meta)
            exp = @call_rule AR(:y, Marginalisation) (q_x = q_x, q_θ = q_θ, q_γ = q_γ, meta = meta)
            @test same_normal(got, exp)
        end
    end
end
