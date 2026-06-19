
@testitem "node:MvNormalGamma average energy" begin
    using ReactiveMP,
        BayesBase, ExponentialFamily, Distributions, LinearAlgebra, StableRNGs

    import ReactiveMP: score, AverageEnergy, Marginal

    avgE(q, μ0, Λ0, α0, β0) = score(
        AverageEnergy(),
        MvNormalGamma,
        Val{(:out, :μ, :Λ, :α, :β)}(),
        (
            Marginal(q, false, false),
            Marginal(PointMass(μ0), false, false),
            Marginal(PointMass(Λ0), false, false),
            Marginal(PointMass(α0), false, false),
            Marginal(PointMass(β0), false, false),
        ),
        nothing,
    )

    @testset "U(prior=q) == entropy(q); U ≥ entropy(q); finite" begin
        rng = StableRNG(88)
        for order in (1, 2, 3)
            A = randn(rng, order, order)
            q = MvNormalGamma(
                randn(rng, order),
                A * A' + diageye(order),
                4.0 + rand(rng),
                2.0 + rand(rng),
            )
            μq, Λq, αq, βq = params(q)

            # Cross-entropy of q against itself is exactly the entropy of q (ties the average
            # energy to the independently-implemented MvNormalGamma entropy).
            @test avgE(q, μq, Λq, αq, βq) ≈ entropy(q)

            B = randn(rng, order, order)
            μ0, Λ0, α0, β0 = randn(rng, order),
            B * B' + diageye(order), 2.0 + rand(rng),
            1.0 + rand(rng)
            U = avgE(q, μ0, Λ0, α0, β0)
            @test isfinite(U)
            @test U ≥ entropy(q) - 1e-8   # cross-entropy ≥ entropy
        end
    end
end
