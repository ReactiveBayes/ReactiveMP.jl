# v6's `test/rules/binomial_polya/{beta_tests,y_tests}.jl`. v6's `BinomialPolyaMeta(k, rng)` is
# `BinomialPolyaApproximation(samples = k)` with the generator in the rule context, seeded as
# v6 seeded its meta. The Monte Carlo checks are v6's, against the mean path within a tolerance.
# v6's node test checked the average energy only, against its plug-in, and `energy_tests.jl`
# replaces it.

@testitem "rules:BinomialPolya:β" tags = [:rules] begin
    using PolyaMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions, StableRNGs
    using MessagePassingRulesBase: RuleContext
    using LinearAlgebra: I, diag

    # The Pólya-Gamma mean at xᵀ⟨β⟩ = 0 is n/4, so Λ = (n/4) x xᵀ and ξ = (y - n/2) x.
    @testset "Multivariate: (q_y::PointMass, q_x::PointMass, q_n::PointMass, m_β::MvNormalWeightedMeanPrecision)" begin
        q = (y = PointMass(3), x = PointMass([0.1, 0.2]), n = PointMass(5))
        m = (β = MvNormalWeightedMeanPrecision(zeros(2), Matrix(1.0I, 2, 2)),)
        Λ = [0.0125 0.025; 0.025 0.05]
        ξ = [0.05, 0.1]

        @test_message_update_rule(
            node = BinomialPolya, target = :β, check_type_promotion = true,
            cases = [(q = q, m = m) => MvNormalWeightedMeanPrecision(ξ, Λ)],
        )

        runs = [
            (BinomialPolyaApproximation(), RuleContext()),
            (BinomialPolyaApproximation(samples = 1), RuleContext(rng = StableRNG(10))),
            (BinomialPolyaApproximation(samples = 10), RuleContext(rng = StableRNG(42))),
        ]
        for (algorithm, ctx) in runs
            out = call_message_update_rule(BinomialPolya, :β; q, m, algorithm, ctx)
            @test weightedmean(out) ≈ ξ rtol = 1.0e-8
            @test diag(precision(out)) ≈ diag(Λ) atol = 1.0e-2
        end
    end

    @testset "Univariate: (q_y::PointMass, q_x::PointMass, q_n::PointMass, m_β::NormalWeightedMeanPrecision)" begin
        q = (y = PointMass(3), x = PointMass(0.1), n = PointMass(5))
        m = (β = NormalWeightedMeanPrecision(0.0, 1.0),)
        Λ = 0.0125
        ξ = 0.05

        @test_message_update_rule(
            node = BinomialPolya, target = :β, check_type_promotion = true,
            cases = [(q = q, m = m) => NormalWeightedMeanPrecision(ξ, Λ)],
        )

        runs = [
            (BinomialPolyaApproximation(), RuleContext()),
            (BinomialPolyaApproximation(samples = 1), RuleContext(rng = StableRNG(10))),
        ]
        for (algorithm, ctx) in runs
            out = call_message_update_rule(BinomialPolya, :β; q, m, algorithm, ctx)
            @test weightedmean(out) ≈ ξ rtol = 1.0e-8
            @test precision(out) ≈ Λ atol = 1.0e-2
        end
    end
end

@testitem "rules:BinomialPolya:y" tags = [:rules] begin
    using PolyaMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions, Random
    using MessagePassingRulesBase: RuleContext
    using LinearAlgebra: I

    # v6 named this item "rules:BinomialPolya:beta" as well.
    @testset "Predictive distribution: (q_x::PointMass, q_n::PointMass, q_β::MvNormalWeightedMeanPrecision)" begin
        q = (x = PointMass([0.1, 0.2]), n = PointMass(5), β = MvNormalWeightedMeanPrecision([3.0, -1.0], Matrix(1.0I, 2, 2)))

        prediction = call_message_update_rule(BinomialPolya, :y; q)
        @test prediction isa Binomial
        @test ntrials(prediction) == 5

        algorithm = BinomialPolyaApproximation(samples = 1000)
        prediction_mc = call_message_update_rule(BinomialPolya, :y; q, algorithm, ctx = RuleContext(rng = MersenneTwister(42)))
        @test prediction_mc isa Binomial
        @test ntrials(prediction_mc) == 5
        @test 0 < succprob(prediction_mc) < 1

        @test succprob(prediction_mc) ≈ succprob(prediction) atol = 1.0e-2
    end
end

@testitem "rules:BinomialPolya:a univariate β, several samples" tags = [:rules] begin
    # v6 drew the samples of a univariate β as a vector and took its one "column", so that
    # `dot(x, column)` threw for more than one sample; v6 only tested one. The draws are the
    # samples themselves now.
    using PolyaMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions, StableRNGs
    using MessagePassingRulesBase: RuleContext

    algorithm = BinomialPolyaApproximation(samples = 10)
    q = (y = PointMass(2.0), x = PointMass(0.5), n = PointMass(4.0))
    mean_path = call_message_update_rule(BinomialPolya, :β; m = (β = NormalMeanVariance(0.3, 0.2),), q)
    sampled = call_message_update_rule(BinomialPolya, :β; m = (β = NormalMeanVariance(0.3, 0.2),), q, algorithm, ctx = RuleContext(rng = StableRNG(1)))
    @test sampled isa NormalWeightedMeanPrecision
    @test weightedmean(sampled) ≈ weightedmean(mean_path)
    @test isfinite(precision(sampled)) && precision(sampled) > 0
    y = call_message_update_rule(BinomialPolya, :y; q = (x = PointMass(0.5), n = PointMass(4.0), β = NormalMeanVariance(0.3, 0.2)), algorithm, ctx = RuleContext(rng = StableRNG(1)))
    @test y isa Binomial && ntrials(y) == 4 && 0 < succprob(y) < 1
end
