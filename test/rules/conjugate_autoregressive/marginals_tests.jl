
@testitem "marginalrules:ConjugateAR:w" begin
    using ReactiveMP,
        BayesBase,
        Random,
        ExponentialFamily,
        Distributions,
        LinearAlgebra,
        StableRNGs

    import ReactiveMP: @call_marginalrule

    # Independent reference implementation of the Bayesian-linear-regression Normal-Gamma
    # posterior (statproofbook.github.io/P/blr-post), used to cross-check the rule.
    function blr_reference(m_w, q_y_x, order)
        myx, Vyx = mean_cov(q_y_x)
        x_idx = (order + 1):(2order)
        mx, my1 = myx[x_idx], myx[1]
        Vx, Vy1, cxy1 = Vyx[x_idx, x_idx], Vyx[1, 1], Vyx[x_idx, 1]
        C = Vx + mx * mx'
        b = cxy1 + mx * my1
        a = Vy1 + my1^2
        μ0, Λ0, α0, β0 = params(m_w)
        Λn = Λ0 + C
        μn = inv(Λn) * (Λ0 * μ0 + b)
        αn = α0 + 1 / 2
        βn = β0 + (a + dot(μ0, Λ0, μ0) - dot(μn, Λn, μn)) / 2
        return (μn, Λn, αn, βn)
    end

    function params_approx(d::MvNormalGamma, ref; atol = 1e-8)
        μ, Λ, α, β = params(d)
        μr, Λr, αr, βr = ref
        return isapprox(μ, μr; atol = atol) &&
               isapprox(Λ, Λr; atol = atol) &&
               isapprox(α, αr; atol = atol) &&
               isapprox(β, βr; atol = atol)
    end

    @testset "order 1: hand-computed posterior" begin
        meta = ARMeta(Multivariate, 1, ARsafe())
        m_w = MvNormalGamma([0.0], fill(2.0, 1, 1), 1.0, 1.0)
        q_y_x = MvNormalMeanCovariance(ones(2), diageye(2))   # C=[2], b=[1], a=2

        q = @call_marginalrule ConjugateAR(:w) (
            m_w = m_w, q_y_x = q_y_x, meta = meta
        )
        μ, Λ, α, β = params(q)

        @test q isa MvNormalGamma
        @test μ ≈ [0.25]
        @test Λ ≈ fill(4.0, 1, 1)
        @test α ≈ 1.5
        @test β ≈ 1.875
    end

    @testset "matches BLR reference (orders 1, 2, 3)" begin
        rng = StableRNG(1234)
        for order in (1, 2, 3)
            meta = ARMeta(Multivariate, order, ARsafe())
            A = randn(rng, 2order, 2order)
            q_y_x = MvNormalMeanCovariance(
                randn(rng, 2order), A * A' + diageye(2order)
            )
            B = randn(rng, order, order)
            m_w = MvNormalGamma(
                randn(rng, order),
                B * B' + diageye(order),
                2.0 + rand(rng),
                1.0 + rand(rng),
            )

            q = @call_marginalrule ConjugateAR(:w) (
                m_w = m_w, q_y_x = q_y_x, meta = meta
            )
            @test params_approx(q, blr_reference(m_w, q_y_x, order))
        end
    end

    @testset "shape bookkeeping: αn = α0 + 1/2" begin
        for order in (1, 2, 3)
            meta = ARMeta(Multivariate, order, ARsafe())
            q_y_x = MvNormalMeanCovariance(ones(2order), diageye(2order))
            α0 = 3.0
            m_w = MvNormalGamma(zeros(order), diageye(order), α0, 1.0)

            q = @call_marginalrule ConjugateAR(:w) (
                m_w = m_w, q_y_x = q_y_x, meta = meta
            )
            @test shape(q) ≈ α0 + 1 / 2
        end
    end

    @testset "properness: Λn ≻ 0 and βn > 0" begin
        rng = StableRNG(7)
        for order in (1, 2, 3)
            meta = ARMeta(Multivariate, order, ARsafe())
            A = randn(rng, 2order, 2order)
            q_y_x = MvNormalMeanCovariance(
                randn(rng, 2order), A * A' + diageye(2order)
            )
            m_w = MvNormalGamma(zeros(order), diageye(order), 2.0, 1.0)

            q = @call_marginalrule ConjugateAR(:w) (
                m_w = m_w, q_y_x = q_y_x, meta = meta
            )
            _, Λ, _, β = params(q)
            @test isposdef(Λ)
            @test β > 0
        end
    end

    @testset "equivalence to prod(prior, likelihood) (order 1)" begin
        meta = ARMeta(Multivariate, 1, ARsafe())
        m_w = MvNormalGamma([0.0], fill(2.0, 1, 1), 1.0, 1.0)
        q_y_x = MvNormalMeanCovariance(ones(2), diageye(2))

        q = @call_marginalrule ConjugateAR(:w) (
            m_w = m_w, q_y_x = q_y_x, meta = meta
        )

        # Likelihood factor in mean parameters: Λ=C, μ=C⁻¹b, α=3/2−d/2, β=a/2 − ½ bᵀC⁻¹b.
        C, b, a = fill(2.0, 1, 1), [1.0], 2.0
        lik = MvNormalGamma(C \ b, C, 3 / 2 - 1 / 2, a / 2 - dot(b, C \ b) / 2)
        qp = prod(PreserveTypeProd(Distribution), m_w, lik)

        @test params_approx(q, params(qp))
    end

    @testset "univariate NormalGamma reduction (d = 1)" begin
        # At order 1 the posterior must equal the scalar Normal-Gamma update.
        meta = ARMeta(Multivariate, 1, ARsafe())
        m_w = MvNormalGamma([0.0], fill(1.0, 1, 1), 2.0, 3.0)
        q_y_x = MvNormalMeanCovariance([2.0, 1.0], [2.0 0.5; 0.5 3.0])   # C=4, b=2.5, a=6

        q = @call_marginalrule ConjugateAR(:w) (
            m_w = m_w, q_y_x = q_y_x, meta = meta
        )
        μ, Λ, α, β = params(q)

        # λn = 1+4 = 5, μn = 2.5/5 = 0.5, αn = 2.5, βn = 3 + (6 − 0.5²·5)/2 = 5.375
        @test μ ≈ [0.5]
        @test Λ ≈ fill(5.0, 1, 1)
        @test α ≈ 2.5
        @test β ≈ 5.375
    end
end

@testitem "marginalrules:ConjugateAR:y_x (delegates to AR)" begin
    using ReactiveMP,
        BayesBase, ExponentialFamily, Distributions, LinearAlgebra, StableRNGs

    import ReactiveMP: @call_marginalrule, conjugatear_effective_marginals

    same_normal(a, b; atol = 1e-8) =
        isapprox(mean(a), mean(b); atol = atol) &&
        isapprox(cov(a), cov(b); atol = atol)

    @testset "y_x joint marginal equals AR(:y_x) with effective (q_θ, q_γ)" begin
        rng = StableRNG(55)
        for order in (1, 2)
            meta = ARMeta(Multivariate, order, ARsafe())
            B = randn(rng, order, order)
            q_w = MvNormalGamma(
                randn(rng, order),
                B * B' + diageye(order),
                2.0 + rand(rng),
                1.0 + rand(rng),
            )
            q_θ, q_γ = conjugatear_effective_marginals(q_w)
            m_y = MvNormalMeanCovariance(randn(rng, order), diageye(order))
            m_x = MvNormalMeanCovariance(randn(rng, order), diageye(order))

            got = @call_marginalrule ConjugateAR(:y_x) (
                m_y = m_y, m_x = m_x, q_w = q_w, meta = meta
            )
            exp = @call_marginalrule AR(:y_x) (
                m_y = m_y, m_x = m_x, q_θ = q_θ, q_γ = q_γ, meta = meta
            )
            @test same_normal(got, exp)
        end
    end
end
