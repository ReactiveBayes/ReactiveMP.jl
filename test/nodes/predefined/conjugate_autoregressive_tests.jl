
@testitem "node:ConjugateAR effective marginals + average energy" begin
    using ReactiveMP, BayesBase, ExponentialFamily, Distributions, LinearAlgebra, StableRNGs

    import ReactiveMP: conjugatear_effective_marginals, score, AverageEnergy, Marginal

    @testset "effective marginals: E[θ]=μ, E[γ]=α/β, mγ·Vθ = Λ⁻¹" begin
        rng = StableRNG(66)
        for order in (1, 2, 3)
            B = randn(rng, order, order)
            Λ = B * B' + diageye(order)
            μ = randn(rng, order)
            α = 2.0 + rand(rng)
            β = 1.0 + rand(rng)
            q_w = MvNormalGamma(μ, Λ, α, β)

            q_θ, q_γ = conjugatear_effective_marginals(q_w)
            @test mean(q_θ) ≈ μ
            @test mean(q_γ) ≈ α / β
            @test mean(q_γ) * cov(q_θ) ≈ inv(Λ)
        end
    end

    @testset "average energy is finite and equals AR with effective marginals" begin
        rng = StableRNG(77)
        for order in (1, 2)
            meta = ARMeta(Multivariate, order, ARsafe())
            A = randn(rng, 2order, 2order)
            q_y_x = MvNormalMeanCovariance(randn(rng, 2order), A * A' + diageye(2order))
            B = randn(rng, order, order)
            q_w = MvNormalGamma(randn(rng, order), B * B' + diageye(order), 2.0 + rand(rng), 1.0 + rand(rng))
            q_θ, q_γ = conjugatear_effective_marginals(q_w)

            ae_car = score(
                AverageEnergy(), ConjugateAR, Val{(:y_x, :w)}(),
                (Marginal(q_y_x, false, false), Marginal(q_w, false, false)), meta,
            )
            ae_ar = score(
                AverageEnergy(), AR, Val{(:y_x, :θ, :γ)}(),
                (Marginal(q_y_x, false, false), Marginal(q_θ, false, false), Marginal(q_γ, false, false)), meta,
            )
            @test isfinite(ae_car)
            @test ae_car ≈ ae_ar
        end
    end
end
