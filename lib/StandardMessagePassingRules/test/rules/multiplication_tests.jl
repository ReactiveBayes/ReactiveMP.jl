# `*`: tables of cases; the messages without a closed form checked against quadrature
# of their defining integrals, the sampled ones with a StableRNG (never the default generator,
# whose stream changes between Julia versions); the log-scale towards `in`; the correction; and
# the refused non-commuting products.

@testitem "rules:*:tables" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions, LinearAlgebra

    I2 = [1.0 0.0; 0.0 1.0]
    @test_message_update_rule(
        node = *, target = :out,
        cases = [
            (m = (A = PointMass(2), in = MvNormalMeanCovariance([1, 2], [3 2; 2 6])),) => MvNormalMeanCovariance([2, 4], [12 8; 8 24]),
            (m = (A = PointMass(0.5), in = MvNormalMeanPrecision([2, 4], [3 2; 2 6])),) => MvNormalMeanPrecision([1, 2], [12 8; 8 24]),
            (m = (A = PointMass(0.5), in = MvNormalWeightedMeanPrecision([1, 2], [3 2; 2 6])),) => MvNormalWeightedMeanPrecision([2, 4], [12 8; 8 24]),
            (m = (A = PointMass(2I), in = MvNormalMeanCovariance([1, 2], [3 2; 2 6])),) => MvNormalMeanCovariance([2, 4], [12 8; 8 24]),
            (m = (A = PointMass(0.5I), in = MvNormalMeanPrecision([2, 4], [3 2; 2 6])),) => MvNormalMeanPrecision([1, 2], [12 8; 8 24]),
            (m = (A = PointMass(0.5I), in = MvNormalWeightedMeanPrecision([1, 2], [3 2; 2 6])),) => MvNormalWeightedMeanPrecision([2, 4], [12 8; 8 24]),
            (m = (A = PointMass([1.0 2.0; 0.0 1.0]), in = MvNormalMeanCovariance([1.0, 1.0], I2)),) => MvNormalMeanCovariance([3.0, 1.0], [5.0 2.0; 2.0 1.0]),
            (m = (A = PointMass([2.0, 1.0]), in = NormalMeanVariance(1.0, 2.0)),) => MvNormalMeanCovariance([2.0, 1.0], [8.0 4.0; 4.0 2.0]),
            (m = (A = PointMass(2.0), in = GammaShapeRate(3.0, 4.0)),) => GammaShapeRate(3.0, 2.0),
            (m = (A = PointMass(2.0), in = PointMass(3.0)),) => PointMass(6.0),
            # A scalar commutes, so each case above holds with the factors swapped.
            (m = (A = GammaShapeRate(3.0, 4.0), in = PointMass(2.0)),) => GammaShapeRate(3.0, 2.0),
            (m = (A = NormalMeanVariance(1.0, 2.0), in = PointMass([2.0, 1.0])),) => MvNormalMeanCovariance([2.0, 1.0], [8.0 4.0; 4.0 2.0]),
            (m = (A = MvNormalMeanCovariance([1, 2], [3 2; 2 6]), in = PointMass(2)),) => MvNormalMeanCovariance([2, 4], [12 8; 8 24]),
        ],
    )
    # Towards `:in`, the distributions come in weighted-mean form.
    @test_message_update_rule(
        node = *, target = :in,
        cases = [
            (m = (A = PointMass(2.0), out = MvNormalMeanCovariance([2.0, 4.0], [12.0 8.0; 8.0 24.0])),) => convert(MvNormalWeightedMeanPrecision, MvNormalMeanCovariance([1.0, 2.0], [3.0 2.0; 2.0 6.0])),
            (m = (A = PointMass(0.5), out = MvNormalMeanPrecision([1.0, 2.0], [12.0 8.0; 8.0 24.0])),) => convert(MvNormalWeightedMeanPrecision, MvNormalMeanPrecision([2.0, 4.0], [3.0 2.0; 2.0 6.0])),
            (m = (A = PointMass(2I), out = MvNormalMeanCovariance([2.0, 4.0], [12.0 8.0; 8.0 24.0])),) => convert(MvNormalWeightedMeanPrecision, MvNormalMeanCovariance([1.0, 2.0], [3.0 2.0; 2.0 6.0])),
            (m = (A = PointMass([1.0 2.0; 0.0 1.0]), out = MvNormalMeanCovariance([3.0, 1.0], I2)),) => MvNormalWeightedMeanPrecision([3.0, 7.0], [1.0 2.0; 2.0 5.0]),
            (m = (A = PointMass([2.0, 1.0]), out = MvNormalMeanPrecision([2.0, 1.0], I2)),) => NormalWeightedMeanPrecision(5.0, 5.0),
            (m = (A = PointMass(2.0), out = GammaShapeRate(3.0, 2.0)),) => GammaShapeRate(3.0, 4.0),
            (m = (A = PointMass([1.0 2.0; 0.0 1.0]), out = PointMass([3.0, 1.0])),) => PointMass([1.0, 1.0]),
        ],
    )
    @test_message_update_rule(
        node = *, target = :A,
        cases = [
            (m = (in = PointMass(2.0), out = NormalMeanVariance(4.0, 8.0)),) => NormalWeightedMeanPrecision(1.0, 0.5),
            (m = (in = PointMass([2.0, 1.0]), out = MvNormalMeanPrecision([2.0, 1.0], I2)),) => NormalWeightedMeanPrecision(5.0, 5.0),
            (m = (in = PointMass(2.0), out = PointMass(6.0)),) => PointMass(3.0),
            (m = (in = PointMass(2.0), out = GammaShapeRate(3.0, 2.0)),) => GammaShapeRate(3.0, 4.0),
        ],
    )
    @test_marginal_update_rule(
        node = *, target = (:A, :in),
        cases = [
            (m = (out = NormalMeanPrecision(0.0, 1.0), A = PointMass(1.0), in = NormalMeanPrecision(1.0, 2.0)),) => FactorizedCluster((:A,) => PointMass(1.0), (:in,) => NormalWeightedMeanPrecision(2.0, 3.0)),
            (m = (out = NormalMeanPrecision(1.0, 2.0), A = PointMass(1.0), in = NormalMeanPrecision(2.0, 1.0)),) => FactorizedCluster((:A,) => PointMass(1.0), (:in,) => NormalWeightedMeanPrecision(4.0, 3.0)),
            (m = (out = NormalMeanPrecision(0.0, 2.0), A = PointMass(2.0), in = NormalMeanPrecision(1.0, 1.0)),) => FactorizedCluster((:A,) => PointMass(2.0), (:in,) => NormalWeightedMeanPrecision(1.0, 9.0)),
            (m = (out = NormalMeanPrecision(0.0, 1.0), A = NormalMeanPrecision(1.0, 2.0), in = PointMass(1.0)),) => FactorizedCluster((:A,) => NormalWeightedMeanPrecision(2.0, 3.0), (:in,) => PointMass(1.0)),
            (m = (out = NormalMeanPrecision(1.0, 2.0), A = NormalMeanPrecision(2.0, 1.0), in = PointMass(1.0)),) => FactorizedCluster((:A,) => NormalWeightedMeanPrecision(4.0, 3.0), (:in,) => PointMass(1.0)),
            (m = (out = NormalMeanPrecision(0.0, 2.0), A = NormalMeanPrecision(1.0, 1.0), in = PointMass(2.0)),) => FactorizedCluster((:A,) => NormalWeightedMeanPrecision(1.0, 9.0), (:in,) => PointMass(2.0)),
            (m = (out = MvNormalMeanPrecision([0.0], [1.0;;]), A = NormalMeanPrecision(0.0, 1.0), in = PointMass([1.0])),) => FactorizedCluster((:A,) => NormalWeightedMeanPrecision(0.0, 2.0), (:in,) => PointMass([1.0])),
            (m = (out = MvNormalMeanPrecision([1.0], [2.0;;]), A = NormalMeanPrecision(2.0, 1.0), in = PointMass([0.0])),) => FactorizedCluster((:A,) => NormalWeightedMeanPrecision(2.0, 1.0), (:in,) => PointMass([0.0])),
            (m = (out = MvNormalMeanPrecision([2.0], [1.0;;]), A = NormalMeanPrecision(1.0, 2.0), in = PointMass([2.0])),) => FactorizedCluster((:A,) => NormalWeightedMeanPrecision(6.0, 6.0), (:in,) => PointMass([2.0])),
        ],
    )
end

@testitem "rules:*:closures" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    using MessagePassingRulesBase: RuleContext, default_algorithm
    using HCubature: hquadrature
    using StableRNGs: StableRNG

    integral(f, lo, hi) = first(hquadrature(f, lo, hi; rtol = 1.0e-10))
    m_out, m_y = NormalMeanVariance(1.5, 0.5), NormalMeanVariance(0.8, 0.3)
    # Towards one factor of out = x·y: ∫ p_out(x y) p_y(y) dy.
    towards_factor(x) = log(integral(y -> pdf(m_out, x * y) * pdf(m_y, y), -8.0, 10.0))
    for target in (:A, :in)
        other = target === :A ? :in : :A
        message = getresult(call_message_update_rule(*, target; m = NamedTuple{(:out, other)}((m_out, m_y))))
        @test message isa ContinuousUnivariateLogPdf
        for x in (-2.0, -0.5, 0.7, 1.9)
            @test logpdf(message, x) ≈ towards_factor(x) rtol = 1.0e-8
        end
    end
    # Towards out: ∫ p_A(a) p_in(z / a) / |a| da, split at a = 0; the Bessel series is truncated.
    m_A, m_in = NormalMeanVariance(1.0, 0.5), NormalMeanVariance(0.5, 0.4)
    product = getresult(call_message_update_rule(*, :out; m = (A = m_A, in = m_in)))
    towards_out(z) = log(integral(a -> pdf(m_A, a) * pdf(m_in, z / a) / abs(a), -8.0, 0.0) + integral(a -> pdf(m_A, a) * pdf(m_in, z / a) / abs(a), 0.0, 8.0))
    for z in (-1.0, 0.3, 1.2, 2.5)
        @test logpdf(product, z) ≈ towards_out(z) rtol = 1.0e-3
    end

    # Any other two univariate distributions: 3000 draws, whose sum is left unnormalised, so
    # the log-density is log 3000 above the integral's, up to the draws' error.
    ctx = RuleContext(rng = StableRNG(42))
    g, b = GammaShapeRate(3.0, 2.0), Beta(2.0, 3.0)
    sampled_in = getresult(call_message_update_rule(*, :in; m = (out = g, A = b), ctx))
    for x in (0.5, 2.0, 4.0)
        @test logpdf(sampled_in, x) - log(3000) ≈ log(integral(y -> pdf(g, x * y) * pdf(b, y), 0.0, 1.0)) atol = 0.05
    end
    sampled_out = getresult(call_message_update_rule(*, :out; m = (A = b, in = g), ctx))
    for z in (0.3, 1.0, 2.0)
        @test logpdf(sampled_out, z) - log(3000) ≈ log(integral(a -> pdf(b, a) * pdf(g, z / a) / a, 0.0, 1.0)) atol = 0.05
    end
    # Towards `A`, the same ratio with the roles of `A` and `in` exchanged.
    sampled_A = getresult(call_message_update_rule(*, :A; m = (out = g, in = b), ctx))
    for a in (0.5, 2.0, 4.0)
        @test logpdf(sampled_A, a) - log(3000) ≈ log(integral(y -> pdf(g, a * y) * pdf(b, y), 0.0, 1.0)) atol = 0.05
    end

    # The number of draws is the algorithm's, 3000 by default.
    @test default_algorithm(*) === MultiplicationSampling() && MultiplicationSampling().samples == 3000
    algorithm = MultiplicationSampling(samples = 5)
    few = getresult(call_message_update_rule(*, :in; m = (out = g, A = b), ctx = RuleContext(rng = StableRNG(1)), algorithm))
    ys = rand(StableRNG(1), b, 5)
    @test logpdf(few, 2.0) ≈ log(sum(y -> pdf(g, 2.0 * y), ys))
end

@testitem "rules:*:logscale-correction-refusals" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesBase, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions
    using MessagePassingRulesBase: RuleNotFoundError, RuleContext
    using MatrixCorrectionTools: NoCorrection
    using BayesBase: tiny

    # m(x) = N_out(a x) integrates to |a|^(-d): the log-scale is -d log |a|, for a < 0 too.
    for (a, out) in ((-2.0, NormalMeanVariance(1.0, 1.0)), (-2.0, MvNormalMeanCovariance([1.0, 2.0], [1.0 0.0; 0.0 1.0])), (0.5, MvNormalMeanCovariance([1.0, 2.0, 3.0], [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0])))
        @test getlogscale(call_message_update_rule(*, :in; m = (out = out, A = PointMass(a)))) ≈ -length(mean(out)) * log(abs(a))
    end
    # A zero column in A gives A'WA a zero on its diagonal: replaced by default, kept by NoCorrection.
    m = (out = MvNormalMeanPrecision([1.0, 1.0], [1.0 0.0; 0.0 1.0]), A = PointMass([1.0 0.0; 0.0 0.0]))
    @test precision(getresult(call_message_update_rule(*, :in; m))) == [1.0 0.0; 0.0 tiny]
    @test precision(getresult(call_message_update_rule(*, :in; m, ctx = RuleContext(matrix_correction = NoCorrection())))) == [1.0 0.0; 0.0 0.0]
    # A matrix `in` does not commute with A, so in * A is refused.
    @test_throws RuleNotFoundError getresult(call_message_update_rule(*, :A; m = (out = MvNormalMeanPrecision([1.0, 1.0], [1.0 0.0; 0.0 1.0]), in = PointMass([1.0 2.0; 0.0 1.0]))))
    @test_throws RuleNotFoundError getresult(call_message_update_rule(*, :out; m = (A = MvNormalMeanPrecision([1.0, 1.0], [1.0 0.0; 0.0 1.0]), in = PointMass([1.0 2.0; 0.0 1.0]))))
end
