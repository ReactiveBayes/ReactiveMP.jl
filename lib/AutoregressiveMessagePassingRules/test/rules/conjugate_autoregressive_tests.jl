# From v6's `test/rules/conjugate_autoregressive/*` and
# `test/nodes/predefined/conjugate_autoregressive_tests.jl`, with the same StableRNG seeds.

@testmodule ConjugateARTestUtils begin
    using BayesBase, ExponentialFamily, Distributions, LinearAlgebra

    diageye(n) = Matrix{Float64}(I, n, n)

    same_normal(a, b; atol = 1.0e-8) = isapprox(mean(a), mean(b); atol) && isapprox(cov(a), cov(b); atol)

    function params_approx(d::MvNormalGamma, ref; atol = 1.0e-8)
        μ, Λ, α, β = params(d)
        μr, Λr, αr, βr = ref
        return isapprox(μ, μr; atol) && isapprox(Λ, Λr; atol) && isapprox(α, αr; atol) && isapprox(β, βr; atol)
    end

    # The expected sufficient statistics of the AR likelihood under q(y, x): C = ⟨x xᵀ⟩,
    # b = ⟨x y₁⟩, a = ⟨y₁²⟩.
    function statistics(q_y_x, order)
        myx, Vyx = mean_cov(q_y_x)
        x_idx = (order + 1):(2order)
        mx, my1 = myx[x_idx], myx[1]
        Vx, Vy1, cxy1 = Vyx[x_idx, x_idx], Vyx[1, 1], Vyx[x_idx, 1]
        return Vx + mx * mx', cxy1 + mx * my1, Vy1 + my1^2
    end

    # The likelihood factor in mean parameters: Λ = C, μ = C⁻¹b, α = (3 - d)/2, β = (a - bᵀC⁻¹b)/2.
    function likelihood_reference(q_y_x, order)
        C, b, a = statistics(q_y_x, order)
        μ = C \ b
        return (μ, C, (3 - order) / 2, (a - dot(b, μ)) / 2)
    end

    # An independent reference for the Bayesian-linear-regression Normal-Gamma posterior
    # (statproofbook.github.io/P/blr-post).
    function posterior_reference(prior, q_y_x, order)
        C, b, a = statistics(q_y_x, order)
        μ0, Λ0, α0, β0 = params(prior)
        Λn = Λ0 + C
        μn = inv(Λn) * (Λ0 * μ0 + b)
        αn = α0 + 1 / 2
        βn = β0 + (a + dot(μ0, Λ0, μ0) - dot(μn, Λn, μn)) / 2
        return (μn, Λn, αn, βn)
    end

    random_joint(rng, order) = (A = randn(rng, 2order, 2order); MvNormalMeanCovariance(randn(rng, 2order), A * A' + diageye(2order)))

    function random_w(rng, order)
        B = randn(rng, order, order)
        return MvNormalGamma(randn(rng, order), B * B' + diageye(order), 2.0 + rand(rng), 1.0 + rand(rng))
    end
end

@testitem "rules:ConjugateAR:w" tags = [:rules] setup = [ConjugateARTestUtils] begin
    using AutoregressiveMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, LinearAlgebra, StableRNGs
    using BayesBase: PreserveTypeProd
    using .ConjugateARTestUtils: diageye, params_approx, likelihood_reference, posterior_reference, random_joint, random_w

    towards_w(q_y_x, order) = call_message_update_rule(ConjugateAR, :w; clusters = ((:y, :x) => q_y_x,), algorithm = ARVMP(Multivariate, order, ARsafe()))

    @testset "likelihood factor parameters (orders 1, 2)" begin
        rng = StableRNG(11)
        for order in (1, 2)
            q_y_x = random_joint(rng, order)
            msg = towards_w(q_y_x, order)
            @test msg isa MvNormalGamma
            @test params_approx(msg, likelihood_reference(q_y_x, order))
        end
    end

    # v6's marginal rule over `w` alone is not ported: the marginal q(w) is the product of the
    # prior with this message, so its tests are ported onto that product.
    posterior(prior, q_y_x, order) = prod(PreserveTypeProd(Distribution), prior, towards_w(q_y_x, order))

    @testset "prod(prior, message) matches the BLR reference (orders 1, 2)" begin
        # v6's `prod(prior, message) == :w marginal posterior`, StableRNG(22).
        rng = StableRNG(22)
        for order in (1, 2)
            q_y_x = random_joint(rng, order)
            prior = random_w(rng, order)
            @test params_approx(posterior(prior, q_y_x, order), posterior_reference(prior, q_y_x, order))
        end
    end

    @testset "order 1: hand-computed posterior" begin
        prior = MvNormalGamma([0.0], fill(2.0, 1, 1), 1.0, 1.0)
        q_y_x = MvNormalMeanCovariance(ones(2), diageye(2))   # C = [2], b = [1], a = 2
        μ, Λ, α, β = params(posterior(prior, q_y_x, 1))
        @test μ ≈ [0.25]
        @test Λ ≈ fill(4.0, 1, 1)
        @test α ≈ 1.5
        @test β ≈ 1.875
    end

    @testset "matches the BLR reference (orders 1, 2, 3)" begin
        rng = StableRNG(1234)
        for order in (1, 2, 3)
            q_y_x = random_joint(rng, order)
            prior = random_w(rng, order)
            @test params_approx(posterior(prior, q_y_x, order), posterior_reference(prior, q_y_x, order))
        end
    end

    @testset "shape bookkeeping: αn = α0 + 1/2" begin
        for order in (1, 2, 3)
            q_y_x = MvNormalMeanCovariance(ones(2order), diageye(2order))
            α0 = 3.0
            prior = MvNormalGamma(zeros(order), diageye(order), α0, 1.0)
            @test shape(posterior(prior, q_y_x, order)) ≈ α0 + 1 / 2
        end
    end

    @testset "properness: Λn ≻ 0 and βn > 0" begin
        rng = StableRNG(7)
        for order in (1, 2, 3)
            q_y_x = random_joint(rng, order)
            prior = MvNormalGamma(zeros(order), diageye(order), 2.0, 1.0)
            _, Λ, _, β = params(posterior(prior, q_y_x, order))
            @test isposdef(Λ)
            @test β > 0
        end
    end

    @testset "univariate NormalGamma reduction (d = 1)" begin
        prior = MvNormalGamma([0.0], fill(1.0, 1, 1), 2.0, 3.0)
        q_y_x = MvNormalMeanCovariance([2.0, 1.0], [2.0 0.5; 0.5 3.0])   # C = 4, b = 2.5, a = 6
        μ, Λ, α, β = params(posterior(prior, q_y_x, 1))
        # λn = 1 + 4 = 5, μn = 2.5 / 5 = 0.5, αn = 2.5, βn = 3 + (6 - 0.5² ⋅ 5) / 2 = 5.375
        @test μ ≈ [0.5]
        @test Λ ≈ fill(5.0, 1, 1)
        @test α ≈ 2.5
        @test β ≈ 5.375
    end
end

@testitem "rules:ConjugateAR:y and x (as AR's)" tags = [:rules] setup = [ConjugateARTestUtils] begin
    using AutoregressiveMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, LinearAlgebra, StableRNGs
    using AutoregressiveMessagePassingRules: conjugatear_effective_marginals
    using .ConjugateARTestUtils: diageye, same_normal, random_w

    # Each rule equals AR's with the marginals of θ and γ that q(w) implies.
    for (target, other, kind, seed) in ((:y, :x, :m, 33), (:y, :x, :q, 34), (:x, :y, :m, 43), (:x, :y, :q, 44))
        @testset "towards $target from $kind[$other]" begin
            rng = StableRNG(seed)
            for order in (1, 2)
                algorithm = ARVMP(Multivariate, order, ARsafe())
                q_w = random_w(rng, order)
                q_θ, q_γ = conjugatear_effective_marginals(q_w)
                input = (; other => MvNormalMeanCovariance(randn(rng, order), diageye(order)))
                got, expected = if kind === :m
                    (
                        call_message_update_rule(ConjugateAR, target; m = input, q = (w = q_w,), algorithm),
                        call_message_update_rule(AR, target; m = input, q = (θ = q_θ, γ = q_γ), algorithm),
                    )
                else
                    (
                        call_message_update_rule(ConjugateAR, target; q = (; input..., w = q_w), algorithm),
                        call_message_update_rule(AR, target; q = (; input..., θ = q_θ, γ = q_γ), algorithm),
                    )
                end
                @test same_normal(got, expected)
            end
        end
    end
end

@testitem "rules:ConjugateAR:marginals (as AR's)" tags = [:rules] setup = [ConjugateARTestUtils] begin
    using AutoregressiveMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, LinearAlgebra, StableRNGs
    using AutoregressiveMessagePassingRules: conjugatear_effective_marginals
    using .ConjugateARTestUtils: diageye, same_normal, random_w

    rng = StableRNG(55)
    for order in (1, 2)
        algorithm = ARVMP(Multivariate, order, ARsafe())
        q_w = random_w(rng, order)
        q_θ, q_γ = conjugatear_effective_marginals(q_w)
        m_y = MvNormalMeanCovariance(randn(rng, order), diageye(order))
        m_x = MvNormalMeanCovariance(randn(rng, order), diageye(order))

        got = call_marginal_update_rule(ConjugateAR, (:y, :x); m = (y = m_y, x = m_x), q = (w = q_w,), algorithm)
        expected = call_marginal_update_rule(AR, (:y, :x); m = (y = m_y, x = m_x), q = (θ = q_θ, γ = q_γ), algorithm)
        @test same_normal(got, expected)
    end
end

@testitem "ConjugateAR: effective marginals and average energy" tags = [:rules] setup = [ConjugateARTestUtils] begin
    using AutoregressiveMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, LinearAlgebra, StableRNGs
    using AutoregressiveMessagePassingRules: conjugatear_effective_marginals
    using .ConjugateARTestUtils: diageye, random_joint, random_w

    @testset "effective marginals: E[θ] = μ, E[γ] = α/β, mγ ⋅ Vθ = Λ⁻¹" begin
        rng = StableRNG(66)
        for order in (1, 2, 3)
            B = randn(rng, order, order)
            Λ = B * B' + diageye(order)
            μ = randn(rng, order)
            α = 2.0 + rand(rng)
            β = 1.0 + rand(rng)
            q_θ, q_γ = conjugatear_effective_marginals(MvNormalGamma(μ, Λ, α, β))
            @test mean(q_θ) ≈ μ
            @test mean(q_γ) ≈ α / β
            @test mean(q_γ) * cov(q_θ) ≈ inv(Λ)
        end
    end

    @testset "average energy is finite and equals AR's with the effective marginals" begin
        rng = StableRNG(77)
        for order in (1, 2)
            algorithm = ARVMP(Multivariate, order, ARsafe())
            q_y_x = random_joint(rng, order)
            q_w = random_w(rng, order)
            q_θ, q_γ = conjugatear_effective_marginals(q_w)

            ae_car = call_average_energy(ConjugateAR; clusters = ((:y, :x) => q_y_x,), q = (w = q_w,), algorithm)
            ae_ar = call_average_energy(AR; clusters = ((:y, :x) => q_y_x,), q = (θ = q_θ, γ = q_γ), algorithm)
            @test isfinite(ae_car)
            @test ae_car ≈ ae_ar
        end
    end
end
