module GCVNodeTest

using Test, ReactiveMP, Distributions, Random, DomainIntegrals, BayesBase, ExponentialFamily

import ReactiveMP: ExponentialLinearQuadratic

@testset "ExponentialLinearQuadratic" begin
    @testset "Statistics" begin
        approximation = GaussHermiteCubature(11)
        rng = MersenneTwister(123)

        for _ in 1:3
            a, b, c, d = rand(rng, 4)
            dist = ExponentialLinearQuadratic(approximation, a, b, c, d)

            μ, v = ReactiveMP.approximate_meancov(approximation, (x) -> exp(-(a * x + b * exp(c * x + d * x^2 / 2)) / 2) * exp(x^2 / 2), NormalMeanVariance(0.0, 1.0))

            σ = sqrt(v)
            w = inv(v)
            ξ = w * μ

            @test mean(dist) ≈ μ
            @test var(dist) ≈ v
            @test cov(dist) ≈ v
            @test std(dist) ≈ σ
            @test weightedmean(dist) ≈ ξ
            @test invcov(dist) ≈ w
            @test precision(dist) ≈ w

            @test all(mean_var(dist) .≈ (μ, v))
            @test all(mean_cov(dist) .≈ (μ, v))
            @test all(mean_invcov(dist) .≈ (μ, w))
            @test all(mean_precision(dist) .≈ (μ, w))
            @test all(mean_std(dist) .≈ (μ, σ))

            @test all(weightedmean_var(dist) .≈ (ξ, v))
            @test all(weightedmean_cov(dist) .≈ (ξ, v))
            @test all(weightedmean_invcov(dist) .≈ (ξ, w))
            @test all(weightedmean_precision(dist) .≈ (ξ, w))
            @test all(weightedmean_std(dist) .≈ (ξ, σ))
        end
    end

    @testset "prod" begin
        approximation = GaussHermiteCubature(51)
        rng = MersenneTwister(1234)

        for _ in 1:3
            a, b, c, d = rand(rng, 4)
            left = ExponentialLinearQuadratic(approximation, a, b, c, d)

            μ, v = 5.0 * randn(rng), 5.0 * rand(rng)
            right = NormalMeanVariance(μ, v)

            q = NormalMeanVariance(ReactiveMP.approximate_meancov(approximation, (x) -> exp(logpdf(left, x) + logpdf(right, x) + x^2 / 2), NormalMeanVariance(0.0, 1.0))...)

            @test all(isapprox.(mean_var(q), mean_var(prod(GenericProd(), left, right)), atol = 1e-2))
            @test all(isapprox.(mean_var(q), mean_var(prod(GenericProd(), right, left)), atol = 1e-2))
        end
    end
end

end
