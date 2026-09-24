# Gauss–Hermite cubature and `approximate_meancov`. v6 tested the cubature only through the GCV
# node and the buffer test below; the exactness cases are new.

@testitem "Gauss–Hermite: exact on polynomials of degree 2p - 1" tags = [:approximations] begin
    using MessagePassingRulesApproximations

    # Under N(m, v), E[x] = m, E[x²] = m² + v, E[x³] = m³ + 3mv, E[x⁴] = m⁴ + 6m²v + 3v².
    m, v = 0.5, 2.0
    moments = (1.0, m, m^2 + v, m^3 + 3m * v, m^4 + 6m^2 * v + 3v^2)
    gh = GaussHermiteCubature(3)
    for k in 0:4
        @test sum(w * x^k for (w, x) in zip(getweights(gh, m, v), getpoints(gh, m, v))) ≈ moments[k + 1]
    end

    # Two dimensions: the weights sum to one, and reproduce the mean and covariance.
    μ, Σ = [1.0, -2.0], [2.0 0.5; 0.5 1.0]
    gh = GaussHermiteCubature(4)
    weights, points = collect(getweights(gh, μ, Σ)), map(copy, getpoints(gh, μ, Σ))
    @test sum(weights) ≈ 1.0
    @test sum(weights .* points) ≈ μ
    @test sum(w * (x - μ) * (x - μ)' for (w, x) in zip(weights, points)) ≈ Σ

    @test ghcubature(5) isa GaussHermiteCubature
    @test length(ghcubature(5).piter) == 5
    @test approximation_name(GaussHermiteCubature(5)) == "GaussHermite(5)"
    @test approximation_short_name(GaussHermiteCubature(5)) == "GH5"
end

@testitem "approximate_meancov: the moments of g(x) N(x | m, v)" tags = [:approximations] begin
    using MessagePassingRulesApproximations

    # g ≡ 1 leaves the normal as it is.
    @test all(approximate_meancov(GaussHermiteCubature(3), x -> 1.0, 0.5, 2.0) .≈ (0.5, 2.0))

    # g(x) = x² under N(0, 1): the mean is E[x³]/E[x²] = 0 and the variance E[x⁴]/E[x²] = 3,
    # exact with three points, since the integrands have degree at most four.
    mean, var = approximate_meancov(GaussHermiteCubature(3), x -> x^2, 0.0, 1.0)
    @test abs(mean) < 1.0e-12 && var ≈ 3.0

    # A normal likelihood N(a, s) times N(m, v) is normal, with precision 1/v + 1/s.
    m, v, a, s = 1.0, 2.0, -0.5, 0.8
    mean, var = approximate_meancov(GaussHermiteCubature(40), x -> exp(-(x - a)^2 / 2s), m, v)
    @test var ≈ 1 / (1 / v + 1 / s) atol = 1.0e-8
    @test mean ≈ var * (m / v + a / s) atol = 1.0e-8

    # The same in two dimensions, with an isotropic likelihood.
    μ, Σ, b = [1.0, 0.0], [2.0 0.3; 0.3 1.0], [0.0, 1.0]
    mean, cov = approximate_meancov(GaussHermiteCubature(30), x -> exp(-sum(abs2, x - b) / 2), μ, Σ)
    P = inv(inv(Σ) + [1.0 0.0; 0.0 1.0])
    @test cov ≈ P atol = 1.0e-6
    @test mean ≈ P * (Σ \ μ + b) atol = 1.0e-6
end

# v6's `test/approximations/getpoints_tests.jl`, its Gauss–Hermite half: the spherical-radial
# cubature it also covered is deleted. The multivariate points are one buffer, rewritten on every
# iteration (ReactiveMP.jl#633): `approximate_meancov` mutates each point in place, which is
# safe only because the next iteration rewrites it. These tests pin that contract.
@testitem "Gauss–Hermite: the multivariate points reuse one buffer, by design" tags = [:approximations] begin
    using MessagePassingRulesApproximations

    μ, Σ = [1.0, 2.0], [1.0 0.2; 0.2 2.0]

    # `collect` yields the same object repeatedly, the last point; do not do this.
    collected = collect(getpoints(GaussHermiteCubature(3), μ, Σ))
    @test length(collected) > 1
    @test all(p -> p === collected[end], collected)

    # `map(copy, …)` is how to materialise them.
    copied = map(copy, getpoints(GaussHermiteCubature(3), μ, Σ))
    @test length(unique(copied)) == length(copied)

    # Consumed lazily, as every caller does, the points are right.
    m̂ = zeros(2)
    for (w, p) in zip(getweights(GaussHermiteCubature(21), μ, Σ), getpoints(GaussHermiteCubature(21), μ, Σ))
        m̂ .+= w .* p
    end
    @test m̂ ≈ μ atol = 1.0e-8

    # The univariate generator yields fresh scalars, so `collect` is safe there.
    @test allunique(collect(getpoints(GaussHermiteCubature(3), 1.0, 2.0)))

    # `approximate_meancov`, which mutates the points, is exact for g ≡ 1.
    m, P = approximate_meancov(GaussHermiteCubature(21), x -> 1.0, μ, Σ)
    @test m ≈ μ atol = 1.0e-8
    @test P ≈ Σ atol = 1.0e-8
end
