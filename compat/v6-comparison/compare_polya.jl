# PolyaMessagePassingRules compared with v6's BinomialPolya and MultinomialPolya on identical
# inputs: the rules on their mean paths, which are deterministic, the Monte Carlo paths as
# distributions, and the energies. Two corrections are declared: BinomialPolya's energy, which v6
# always took at the mean of xᵀβ, and MultinomialPolya's for a Multinomial q(x) with N > 1.
#
#   julia --startup-file=no --project=compat/v6-comparison compat/v6-comparison/compare_polya.jl

include(joinpath(@__DIR__, "V6Oracle.jl"))

using .V6Oracle, Test, Random
using ExponentialFamily, BayesBase, Distributions, LinearAlgebra
using MessagePassingRulesBase, MessagePassingRulesTestUtils, PolyaMessagePassingRules
using MessagePassingRulesBase: RuleContext
import ReactiveMP

const BINOMIAL_ENERGY = """
v6's BinomialPolya energy took softplus at the mean of ψ = xᵀβ, and overwrote the Monte Carlo
estimate it computed with that plug-in, so it was biased low whatever the meta. The port computes
⟨softplus(ψ)⟩ under the normal of ψ by Gauss–Hermite cubature (user, 2026-09-24).
"""

const MULTINOMIAL_ENERGY = """
v6's MultinomialPolya energy, for a Multinomial q(x), took ⟨log C(x)⟩ as Σ_k ⟨log x_k!⟩ - log N,
where it is log N! - Σ_k ⟨log x_k!⟩: the sign flipped and log N for log N!. It was right for N = 1
and for observed counts. The package's energy tests pin the port against enumeration.
"""

@testset "PolyaMessagePassingRules against v6" begin
    @testset "BinomialPolya" begin
        for (x, y, n) in (([1.0, -0.5], 3.0, 5.0), ([0.2, 0.7, -1.1], 0.0, 4.0), ([2.0], 7.0, 7.0))
            d = length(x)
            m_β = MvNormalMeanCovariance(fill(0.3, d), Matrix(Diagonal(fill(0.5, d))))
            q_β = MvNormalMeanCovariance(fill(-0.2, d), Matrix(Diagonal(fill(0.8, d))))
            q = (y = PointMass(y), x = PointMass(x), n = PointMass(n))
            label = "x = $x, y = $y, n = $n"

            # The mean paths.
            v7 = getresult(call_message_update_rule(BinomialPolya, :β; m = (β = m_β,), q))
            v6, _ = v6_message_update(ReactiveMP.BinomialPolya, :β, (β = m_β,), q)
            @test compare_with_reference("BinomialPolya:β:$label", v7, v6; node = "BinomialPolya", target = ":β").outcome === :agree
            v7 = getresult(call_message_update_rule(BinomialPolya, :y; q = (x = q.x, n = q.n, β = q_β)))
            v6, _ = v6_message_update(ReactiveMP.BinomialPolya, :y, NamedTuple(), (x = q.x, n = q.n, β = q_β))
            @test compare_with_reference("BinomialPolya:y:$label", v7, v6; node = "BinomialPolya", target = ":y").outcome === :agree

            # The Monte Carlo paths draw from different generators, so they are compared as
            # distributions: with many samples, each within its sampling error of the other.
            samples = 20_000
            algorithm, ctx = BinomialPolyaApproximation(; samples), RuleContext(rng = Xoshiro(7))
            meta = ReactiveMP.BinomialPolyaMeta(samples, Xoshiro(11))
            v7 = getresult(call_message_update_rule(BinomialPolya, :β; m = (β = m_β,), q, algorithm, ctx))
            v6, _ = v6_message_update(ReactiveMP.BinomialPolya, :β, (β = m_β,), q; meta)
            @test compare_with_reference("BinomialPolya:β:MC:$label", v7, v6; node = "BinomialPolya", target = ":β", rtol = 2.0e-2).outcome === :agree
            v7 = getresult(call_message_update_rule(BinomialPolya, :y; q = (x = q.x, n = q.n, β = q_β), algorithm, ctx))
            v6, _ = v6_message_update(ReactiveMP.BinomialPolya, :y, NamedTuple(), (x = q.x, n = q.n, β = q_β); meta)
            @test compare_with_reference("BinomialPolya:y:MC:$label", v7, v6; node = "BinomialPolya", target = ":y", atol = 1.0e-2).outcome === :agree

            # The energy: corrected for a normal q(β), and the same for a point mass, where the
            # plug-in is the expectation.
            id = "BinomialPolya:energy:$label"
            v7 = getresult(call_average_energy(BinomialPolya; q = merge(q, (β = q_β,))))
            v6 = v6_average_energy(ReactiveMP.BinomialPolya, merge(q, (β = q_β,)))
            declared = [DeclaredDisagreement(id; kind = :correction, reasoning = BINOMIAL_ENERGY)]
            @test compare_with_reference(id, v7, v6; node = "BinomialPolya", target = "energy", declared).outcome === :correction
            v7 = getresult(call_average_energy(BinomialPolya; q = merge(q, (β = PointMass(mean(q_β)),))))
            v6 = v6_average_energy(ReactiveMP.BinomialPolya, merge(q, (β = PointMass(mean(q_β)),)))
            @test compare_with_reference("BinomialPolya:energy:point mass:$label", v7, v6; node = "BinomialPolya", target = "energy").outcome === :agree
        end
    end

    @testset "MultinomialPolya" begin
        meta = ReactiveMP.MultinomialPolyaMeta(21)
        for (counts, N) in (([3, 2, 5], 10), ([1, 0, 0, 0], 1), ([4, 6], 10))
            K = length(counts)
            m_ψ = MvNormalMeanCovariance(fill(0.2, K - 1), Matrix(Diagonal(fill(0.6, K - 1))))
            q_ψ = MvNormalMeanCovariance([-0.3 + 0.7 * (k - 1) / max(K - 2, 1) for k in 1:(K - 1)], Matrix(Diagonal(fill(0.4, K - 1))))
            label = "x = $counts"

            v7 = getresult(call_message_update_rule(MultinomialPolya, :ψ; m = (ψ = m_ψ,), q = (x = PointMass(counts), N = PointMass(N))))
            v6, _ = v6_message_update(ReactiveMP.MultinomialPolya, :ψ, (ψ = m_ψ,), (x = PointMass(counts), N = PointMass(N)); meta)
            @test compare_with_reference("MultinomialPolya:ψ:$label", v7, v6; node = "MultinomialPolya", target = ":ψ").outcome === :agree
            v7 = getresult(call_message_update_rule(MultinomialPolya, :x; q = (N = PointMass(N), ψ = q_ψ)))
            v6, _ = v6_message_update(ReactiveMP.MultinomialPolya, :x, NamedTuple(), (N = PointMass(N), ψ = q_ψ); meta)
            @test compare_with_reference("MultinomialPolya:x:$label", v7, v6; node = "MultinomialPolya", target = ":x").outcome === :agree

            # Observed counts: v6 was right.
            q = (x = PointMass(counts), N = PointMass(N), ψ = q_ψ)
            v7 = getresult(call_average_energy(MultinomialPolya; q))
            v6 = v6_average_energy(ReactiveMP.MultinomialPolya, q; meta)
            @test compare_with_reference("MultinomialPolya:energy:$label", v7, v6; node = "MultinomialPolya", target = "energy").outcome === :agree

            # A Multinomial q(x): right for N = 1, corrected otherwise.
            q = (x = Multinomial(N, fill(1 / K, K)), N = PointMass(N), ψ = q_ψ)
            v7 = getresult(call_average_energy(MultinomialPolya; q))
            v6 = v6_average_energy(ReactiveMP.MultinomialPolya, q; meta)
            id = "MultinomialPolya:energy:Multinomial:$label"
            declared = N == 1 ? DeclaredDisagreement[] : [DeclaredDisagreement(id; kind = :correction, reasoning = MULTINOMIAL_ENERGY)]
            @test compare_with_reference(id, v7, v6; node = "MultinomialPolya", target = "energy", declared).outcome === (N == 1 ? :agree : :correction)
        end
    end
end
