# The average energies, against exact expectations: a fine quadrature of ⟨softplus(ψ)⟩ over the
# normal of ψ, and enumeration over a Multinomial q(x). v6's were wrong in both nodes, checked in
# 6.5.0: BinomialPolya's always took softplus at the mean of ψ, and MultinomialPolya's, for a
# Multinomial q(x), flipped the sign of Σ ⟨log x_k!⟩ and took log N for log N!.

@testmodule PolyaReference begin
    using Distributions, LogExpFunctions, SpecialFunctions

    # ⟨softplus(ψ)⟩ for ψ ~ N(m, v), by a fine rectangle rule over ±10 standard deviations.
    function expected_softplus(m, v; n = 200_001)
        z = range(-10, 10; length = n)
        w = pdf.(Normal(), z)
        return sum(w .* softplus.(m .+ sqrt(v) .* z)) / sum(w)
    end

    # ⟨-log p(x | N, ψ)⟩ over q(x) = Multinomial(N, π) and independent normal ψ_k, by enumeration.
    function multinomial_energy(N, π, μψ, vψ)
        K, q, total = length(π), Multinomial(N, π), 0.0
        for x in Iterators.product(ntuple(_ -> 0:N, K)...)
            sum(x) == N || continue
            Nk = [N - sum(x[1:(k - 1)]; init = 0) for k in 1:(K - 1)]
            energy = -loggamma(N + 1) + sum(loggamma.(collect(x) .+ 1)) -
                sum(x[k] * μψ[k] - Nk[k] * expected_softplus(μψ[k], vψ[k]) for k in 1:(K - 1))
            total += pdf(q, collect(x)) * energy
        end
        return total
    end
end

@testitem "rules:BinomialPolya:energy" tags = [:rules] setup = [PolyaReference] begin
    using PolyaMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions, LinearAlgebra
    using SpecialFunctions: loggamma
    using LogExpFunctions: softplus
    R = PolyaReference

    x, y, n = [1.0, -0.5], 3.0, 5.0
    q_β = MvNormalMeanCovariance([0.4, 0.2], [0.8 0.1; 0.1 0.5])
    mψ, vψ = dot(x, mean(q_β)), dot(x, cov(q_β) * x)
    coefficient = loggamma(n + 1) - loggamma(n - y + 1) - loggamma(y + 1)
    exact = -coefficient - y * mψ + n * R.expected_softplus(mψ, vψ)
    q = (y = PointMass(y), x = PointMass(x), n = PointMass(n), β = q_β)
    # 1.5337, where v6 gave the plug-in 1.0692 whether or not it was asked to sample.
    for algorithm in (BinomialPolyaApproximation(), BinomialPolyaApproximation(samples = 100))
        energy = call_average_energy(BinomialPolya; q, algorithm)
        @test energy ≈ exact rtol = 1.0e-8
        @test energy > -coefficient - y * mψ + n * softplus(mψ)
    end
    # A point-mass q(β) has no spread, and the energy is the negative log-likelihood itself.
    β = [0.3, -0.1]
    @test call_average_energy(BinomialPolya; q = (y = PointMass(y), x = PointMass(x), n = PointMass(n), β = PointMass(β))) ≈
        -logpdf(Binomial(Int(n), 1 / (1 + exp(-dot(x, β)))), Int(y))
end

@testitem "rules:MultinomialPolya:energy" tags = [:rules] setup = [PolyaReference] begin
    using PolyaMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions, LinearAlgebra
    using SpecialFunctions: loggamma
    R = PolyaReference

    μψ, vψ = [0.3, -0.4], [0.5, 0.2]
    q_ψ = MvNormalMeanCovariance(μψ, Matrix(Diagonal(vψ)))
    # v6 gave 1.4016, 4.5592 and 3.9500: right for N = 1 only.
    for (N, π) in ((1, [0.2, 0.3, 0.5]), (3, [0.2, 0.3, 0.5]), (5, [0.6, 0.1, 0.3]))
        energy = call_average_energy(MultinomialPolya; q = (x = Multinomial(N, π), N = PointMass(N), ψ = q_ψ))
        @test energy ≈ R.multinomial_energy(N, π, μψ, vψ) rtol = 1.0e-6
        # Observed counts, v6's branch, which was right.
        x = [N - (N ÷ 2) - (N ÷ 3), N ÷ 2, N ÷ 3]
        Nk = [N, N - x[1]]
        expected = -loggamma(N + 1) + sum(loggamma.(x .+ 1)) - sum(x[k] * μψ[k] - Nk[k] * R.expected_softplus(μψ[k], vψ[k]) for k in 1:2)
        @test call_average_energy(MultinomialPolya; q = (x = PointMass(x), N = PointMass(N), ψ = q_ψ)) ≈ expected rtol = 1.0e-6
    end
    # v6's node test for observed counts and a point-mass ψ, where its energy was right.
    @test call_average_energy(MultinomialPolya; q = (x = PointMass([30, 70]), N = PointMass(100), ψ = PointMass([0.5]))) ≈ 23.76 atol = 0.1
    # The energy of a discrete x is never negative; v6's node test asserted -101.72 here.
    energy = call_average_energy(MultinomialPolya; q = (x = Multinomial(100, [0.2, 0.3, 0.5]), N = PointMass(100), ψ = MvNormalMeanCovariance([0.1, -0.2], [2.0 0.5; 0.5 1.5])))
    @test energy > 0
end
