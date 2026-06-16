
@testitem "rules:ConjugateAR:w (message)" begin
    using ReactiveMP,
        BayesBase, ExponentialFamily, Distributions, LinearAlgebra, StableRNGs

    import ReactiveMP: @call_rule, @call_marginalrule

    # Likelihood factor in mean parameters: Λ = C, μ = C⁻¹b, α = (3−d)/2, β = (a − bᵀC⁻¹b)/2,
    # with C = E[xxᵀ], b = E[x y₁], a = E[y₁²] under q(y, x).
    function lik_reference(q_y_x, order)
        myx, Vyx = mean_cov(q_y_x)
        x_idx = (order + 1):(2order)
        mx, my1 = myx[x_idx], myx[1]
        Vx, Vy1, cxy1 = Vyx[x_idx, x_idx], Vyx[1, 1], Vyx[x_idx, 1]
        C = Vx + mx * mx'
        b = cxy1 + mx * my1
        a = Vy1 + my1^2
        μ = C \ b
        return (μ, C, (3 - order) / 2, (a - dot(b, μ)) / 2)
    end

    function params_approx(d::MvNormalGamma, ref; atol = 1e-8)
        μ, Λ, α, β = params(d)
        μr, Λr, αr, βr = ref
        return isapprox(μ, μr; atol = atol) &&
               isapprox(Λ, Λr; atol = atol) &&
               isapprox(α, αr; atol = atol) &&
               isapprox(β, βr; atol = atol)
    end

    @testset "likelihood factor parameters (orders 1, 2)" begin
        rng = StableRNG(11)
        for order in (1, 2)
            meta = ARMeta(Multivariate, order, ARsafe())
            A = randn(rng, 2order, 2order)
            q_y_x = MvNormalMeanCovariance(
                randn(rng, 2order), A * A' + diageye(2order)
            )

            msg = @call_rule ConjugateAR(:w, Marginalisation) (
                q_y_x = q_y_x, meta = meta
            )
            @test msg isa MvNormalGamma
            @test params_approx(msg, lik_reference(q_y_x, order))
        end
    end

    # The marginal q(w) is the equality-node product of the prior with this message; it must
    # equal the directly-computed conjugate posterior (the :w marginal rule).
    @testset "prod(prior, message) == :w marginal posterior" begin
        rng = StableRNG(22)
        for order in (1, 2)
            meta = ARMeta(Multivariate, order, ARsafe())
            A = randn(rng, 2order, 2order)
            q_y_x = MvNormalMeanCovariance(
                randn(rng, 2order), A * A' + diageye(2order)
            )
            B = randn(rng, order, order)
            prior = MvNormalGamma(
                randn(rng, order),
                B * B' + diageye(order),
                2.0 + rand(rng),
                1.0 + rand(rng),
            )

            msg = @call_rule ConjugateAR(:w, Marginalisation) (
                q_y_x = q_y_x, meta = meta
            )
            post_prod = prod(PreserveTypeProd(Distribution), prior, msg)
            post_marg = @call_marginalrule ConjugateAR(:w) (
                m_w = prior, q_y_x = q_y_x, meta = meta
            )

            @test params_approx(post_prod, params(post_marg))
        end
    end
end
