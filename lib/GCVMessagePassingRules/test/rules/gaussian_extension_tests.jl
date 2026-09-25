# The rules this package adds to NormalMeanVariance and NormalMeanPrecision for an
# ExponentialLinearQuadratic message on `out`. The message is reduced to a
# normal of its moments, so each rule must agree with Standard's own rule applied to that normal.

@testitem "rules:GCV:gaussian extension, messages towards μ" tags = [:rules] setup = [GCVRulesTestUtils] begin
    using GCVMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    using MessagePassingRulesApproximations: GaussHermiteCubature

    elqs = (
        ExponentialLinearQuadratic(GaussHermiteCubature(20), 1.0, 1.0, -1.0, 0.0),
        ExponentialLinearQuadratic(GaussHermiteCubature(20), 0.8, 2.5, -0.8, 0.4),
    )
    for elq in elqs
        normal = NormalMeanVariance(mean_var(elq)...)
        for (node, inputs) in (
                (NormalMeanVariance, ((m = (v = PointMass(2.0),),), (q = (v = InverseGamma(3.0, 4.0),),), (q = (v = PointMass(0.5),),))),
                (NormalMeanPrecision, ((m = (τ = PointMass(2.0),),), (q = (τ = GammaShapeRate(3.0, 2.0),),), (q = (τ = PointMass(0.5),),))),
            )
            for input in inputs
                m, q = get(input, :m, (;)), get(input, :q, (;))
                result = call_message_update_rule(node, :μ; m = (out = elq, m...), q)
                reference = call_message_update_rule(node, :μ; m = (out = normal, m...), q)
                @test result isa NormalMeanVariance
                @test all(mean_var(result) .≈ mean_var(reference))
            end
        end
    end
    # By hand, for the first: the moments' variance plus the variance of `v`, or of 1/τ.
    elq = first(elqs)
    elq_mean, elq_var = mean_var(elq)
    @test all(mean_var(call_message_update_rule(NormalMeanVariance, :μ; m = (out = elq, v = PointMass(2.0)))) .≈ (elq_mean, elq_var + 2.0))
    @test all(mean_var(call_message_update_rule(NormalMeanPrecision, :μ; m = (out = elq,), q = (τ = GammaShapeRate(3.0, 2.0),))) .≈ (elq_mean, elq_var + 2 / 3))
end

@testitem "marginalrules:GCV:gaussian extension, the joint of out and μ" tags = [:rules] setup = [GCVRulesTestUtils] begin
    using GCVMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    using MessagePassingRulesApproximations: GaussHermiteCubature

    elq = ExponentialLinearQuadratic(GaussHermiteCubature(20), 0.8, 2.5, -0.8, 0.4)
    normal = NormalMeanVariance(mean_var(elq)...)
    m_μ = NormalMeanVariance(0.5, 1.5)
    same_joint(a, b) = all(mean_cov(a) .≈ mean_cov(b))

    # The whole node as one cluster, `v` or `τ` a point mass: the blocks `(:out, :μ)` and `(:v,)`.
    for (node, target, param) in ((NormalMeanVariance, (:out, :μ, :v), :v), (NormalMeanPrecision, (:out, :μ, :τ), :τ))
        m_param = NamedTuple{(param,)}((PointMass(2.0),))
        result = call_marginal_update_rule(node, target; m = (out = elq, μ = m_μ, m_param...))
        reference = call_marginal_update_rule(node, target; m = (out = normal, μ = m_μ, m_param...))
        @test result[(:out, :μ)] isa MvNormalWeightedMeanPrecision
        @test same_joint(result[(:out, :μ)], reference[(:out, :μ)])
        @test result[(param,)] == reference[(param,)] == PointMass(2.0)
    end
    # The structured `q(out, μ) q(v)`, or `q(out, μ) q(τ)`.
    for (node, q) in ((NormalMeanVariance, (v = InverseGamma(3.0, 4.0),)), (NormalMeanPrecision, (τ = GammaShapeRate(3.0, 2.0),)))
        result = call_marginal_update_rule(node, (:out, :μ); m = (out = elq, μ = m_μ), q)
        reference = call_marginal_update_rule(node, (:out, :μ); m = (out = normal, μ = m_μ), q)
        @test result isa MvNormalWeightedMeanPrecision
        @test same_joint(result, reference)
    end
    # By hand: the precision couples the two messages' precisions by W̄ = E[1/v] = 3/4.
    xi_out, w_out = weightedmean_precision(normal)
    xi_μ, w_μ = weightedmean_precision(m_μ)
    joint = call_marginal_update_rule(NormalMeanVariance, (:out, :μ); m = (out = elq, μ = m_μ), q = (v = InverseGamma(3.0, 4.0),))
    @test weightedmean(joint) ≈ [xi_out, xi_μ]
    @test invcov(joint) ≈ [w_out + 0.75 -0.75; -0.75 w_μ + 0.75]
end
