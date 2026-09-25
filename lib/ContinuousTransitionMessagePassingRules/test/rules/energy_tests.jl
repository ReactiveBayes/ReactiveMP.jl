# The average energies, ⟨-log N(y; A x, W⁻¹)⟩ = dy/2 log 2π - ⟨log det W⟩/2 + tr(⟨W⟩ E[(y - A x)(y - A x)ᵀ])/2.
# These are checked against the closed form and a Monte Carlo estimate.

@testitem "rules:ContinuousTransition:energy, closed form" tags = [:rules] begin
    using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions, LinearAlgebra
    using SpecialFunctions: digamma

    # dy = 2, dx = 3, A = reshape(a, 2, 3), every mean zero and every covariance the identity,
    # W ~ Wishart(3, I). Then E[(y - A x)(y - A x)ᵀ] = I + E[A Aᵀ] = I + 3 I, since each row of A
    # holds three entries of `a`, and ⟨W⟩ = 3 I, ⟨log det W⟩ = ψ(3/2) + ψ(1) + 2 log 2.
    dy, dx = 2, 3
    algorithm = CTVMP(a -> reshape(a, dy, dx))
    q_y, q_x = MvNormalMeanCovariance(zeros(dy), Matrix(1.0I, dy, dy)), MvNormalMeanCovariance(zeros(dx), Matrix(1.0I, dx, dx))
    q_y_x = MvNormalMeanCovariance(zeros(dy + dx), Matrix(1.0I, dy + dx, dy + dx))
    q_a, q_W = MvNormalMeanCovariance(zeros(dx * dy), Matrix(1.0I, dx * dy, dx * dy)), Wishart(dy + 1, Matrix(1.0I, dy, dy))
    expected = log(2π) - (digamma(3 / 2) + digamma(1) + 2 * log(2)) / 2 + tr(3 * 4 * I(dy)) / 2
    @test expected ≈ 13.415092731310878
    @test getresult(call_average_energy(ContinuousTransition; clusters = ((:y, :x) => q_y_x,), q = (a = q_a, W = q_W), algorithm)) ≈ expected
    @test getresult(call_average_energy(ContinuousTransition; q = (y = q_y, x = q_x, a = q_a, W = q_W), algorithm)) ≈ expected
end

@testitem "rules:ContinuousTransition:energy, against Monte Carlo" tags = [:rules] begin
    using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions, LinearAlgebra, StableRNGs

    spd(d, s) = [i == j ? s + i : 0.2 / (i + j) for i in 1:d, j in 1:d]
    function monte_carlo(f, sample_y_x, q_a, q_W, dy; n = 200_000, rng = StableRNG(42))
        total = 0.0
        for _ in 1:n
            y, x = sample_y_x(rng)
            a, W = rand(rng, q_a), rand(rng, q_W)
            r = y - f(a) * x
            total += dy / 2 * log(2π) - logdet(W) / 2 + dot(r, W * r) / 2
        end
        return total / n
    end

    # A linear f, with dy ≠ dx both ways, so that the dimension term and the Vx term are exercised.
    for (dy, dx) in ((1, 2), (2, 3), (3, 2))
        f = a -> reshape(a, dy, dx)
        algorithm = CTVMP(f)
        q_a = MvNormalMeanCovariance([0.3 - 0.7 * (k - 1) / (dy * dx - 1) for k in 1:(dy * dx)], 0.05 * spd(dy * dx, 1.0))
        q_W = Wishart(dy + 3, spd(dy, 0.5) / 3)
        q_y = MvNormalMeanCovariance([1.0 - 0.8 * (k - 1) / max(dy - 1, 1) for k in 1:dy], 0.3 * spd(dy, 1.0))
        q_x = MvNormalMeanCovariance([-0.5 + 1.3 * (k - 1) / (dx - 1) for k in 1:dx], 0.4 * spd(dx, 1.0))
        q_y_x = MvNormalMeanCovariance(vcat(mean(q_y), mean(q_x)), 0.3 * spd(dy + dx, 1.5))

        meanfield = getresult(call_average_energy(ContinuousTransition; q = (y = q_y, x = q_x, a = q_a, W = q_W), algorithm))
        @test meanfield ≈ monte_carlo(f, rng -> (rand(rng, q_y), rand(rng, q_x)), q_a, q_W, dy) rtol = 5.0e-3
        structured = getresult(call_average_energy(ContinuousTransition; clusters = ((:y, :x) => q_y_x,), q = (a = q_a, W = q_W), algorithm))
        @test structured ≈ monte_carlo(f, rng -> (z = rand(rng, q_y_x); (z[1:dy], z[(dy + 1):end])), q_a, q_W, dy) rtol = 5.0e-3

        # A joint without cross-covariance is the mean-field energy.
        block = MvNormalMeanCovariance(vcat(mean(q_y), mean(q_x)), [cov(q_y) zeros(dy, dx); zeros(dx, dy) cov(q_x)])
        @test getresult(call_average_energy(ContinuousTransition; clusters = ((:y, :x) => block,), q = (a = q_a, W = q_W), algorithm)) ≈ meanfield
    end
end

@testitem "rules:ContinuousTransition:energy, float types" tags = [:rules] begin
    using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions, LinearAlgebra

    # Float32 inputs give a Float32 energy: the log 2π term takes the inputs' type.
    algorithm = CTVMP(a -> reshape(a, 2, 2))
    q_y, q_x = MvNormalMeanCovariance(Float32[1, 0], Matrix{Float32}(I, 2, 2)), MvNormalMeanCovariance(Float32[0, 1], Matrix{Float32}(I, 2, 2))
    q_a, q_W = MvNormalMeanCovariance(Float32[1, 0, 0, 1], Matrix{Float32}(I, 4, 4)), Wishart(3.0f0, Matrix{Float32}(I, 2, 2))
    @test getresult(call_average_energy(ContinuousTransition; q = (y = q_y, x = q_x, a = q_a, W = q_W), algorithm)) isa Float32
end
