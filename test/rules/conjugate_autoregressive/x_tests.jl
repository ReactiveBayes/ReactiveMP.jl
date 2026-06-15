
@testitem "rules:ConjugateAR:x (delegates to AR)" begin
    using ReactiveMP, BayesBase, ExponentialFamily, Distributions, LinearAlgebra, StableRNGs

    import ReactiveMP: @call_rule, conjugatear_effective_marginals

    same_normal(a, b; atol = 1e-8) =
        isapprox(mean(a), mean(b); atol = atol) && isapprox(cov(a), cov(b); atol = atol)

    @testset "x message equals AR(:x) with effective (q_θ, q_γ)" begin
        rng = StableRNG(43)
        for order in (1, 2)
            meta = ARMeta(Multivariate, order, ARsafe())
            B = randn(rng, order, order)
            q_w = MvNormalGamma(randn(rng, order), B * B' + diageye(order), 2.0 + rand(rng), 1.0 + rand(rng))
            q_θ, q_γ = conjugatear_effective_marginals(q_w)
            m_y = MvNormalMeanCovariance(randn(rng, order), diageye(order))

            got = @call_rule ConjugateAR(:x, Marginalisation) (m_y = m_y, q_w = q_w, meta = meta)
            exp = @call_rule AR(:x, Marginalisation) (m_y = m_y, q_θ = q_θ, q_γ = q_γ, meta = meta)
            @test same_normal(got, exp)
        end
    end

    @testset "x message (q_y variant) equals AR(:x)" begin
        rng = StableRNG(44)
        for order in (1, 2)
            meta = ARMeta(Multivariate, order, ARsafe())
            B = randn(rng, order, order)
            q_w = MvNormalGamma(randn(rng, order), B * B' + diageye(order), 2.0 + rand(rng), 1.0 + rand(rng))
            q_θ, q_γ = conjugatear_effective_marginals(q_w)
            q_y = MvNormalMeanCovariance(randn(rng, order), diageye(order))

            got = @call_rule ConjugateAR(:x, Marginalisation) (q_y = q_y, q_w = q_w, meta = meta)
            exp = @call_rule AR(:x, Marginalisation) (q_y = q_y, q_θ = q_θ, q_γ = q_γ, meta = meta)
            @test same_normal(got, exp)
        end
    end
end
