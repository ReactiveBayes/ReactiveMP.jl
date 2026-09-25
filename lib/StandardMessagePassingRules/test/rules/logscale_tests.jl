# Log scales derived for deterministic nodes, checked against their definitions: the log of the
# integral, or the sum, of the unnormalised message each rule stands for.

@testitem "rules:logscale:logic nodes by enumeration" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions

    # The unnormalised message towards one input: the sum over the other variables of the
    # indicator of the node's relation times their messages.
    ber(p, x) = x ? p : 1 - p
    towards_input(f, pout, pother, first) = map((false, true)) do a
        sum(ber(pout, f((first ? (a, b) : (b, a))...)) * ber(pother, b) for b in (false, true))
    end
    nodes = ((AND, (a, b) -> a & b), (OR, (a, b) -> a | b), (IMPLY, (a, b) -> !a | b))
    for (node, f) in nodes, pout in (0.1, 0.35, 0.8), pother in (0.2, 0.5, 0.9)
        for (target, other, first) in ((:in1, :in2, true), (:in2, :in1, false))
            m = towards_input(f, pout, pother, first)
            result = call_message_update_rule(node, target; m = NamedTuple{(:out, other)}((Bernoulli(pout), Bernoulli(pother))))
            @test mean(getresult(result)) ≈ m[2] / sum(m)
            @test getlogscale(result) ≈ log(sum(m))
        end
        # Towards `out`, the pushforward of the inputs: normalised.
        @test getlogscale(call_message_update_rule(node, :out; m = (in1 = Bernoulli(pout), in2 = Bernoulli(pother)))) === 0
    end
    @test getlogscale(call_message_update_rule(NOT, :in; m = (out = Bernoulli(0.3),))) === 0
end

@testitem "rules:logscale:* by quadrature" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    using HCubature: hcubature

    # a ↦ m_out(a c) integrates to 1 / |c|.
    integral(f, lower, upper) = first(hcubature(x -> f(x[1]), (lower,), (upper,); rtol = 1.0e-10))
    out, c = GammaShapeRate(3.0, 2.0), 2.5
    for (target, known) in ((:in, :A), (:A, :in))
        result = call_message_update_rule(*, target; m = NamedTuple{(:out, known)}((out, PointMass(c))))
        @test exp(getlogscale(result)) ≈ integral(x -> pdf(out, c * x), 0.0, 40.0) rtol = 1.0e-8
    end
    normal = NormalMeanVariance(1.0, 2.0)
    result = call_message_update_rule(*, :A; m = (out = normal, in = PointMass(-c)))
    @test exp(getlogscale(result)) ≈ integral(x -> pdf(normal, -c * x), -30.0, 30.0) rtol = 1.0e-8
    # A known scale's product with a Gamma is its pushforward: normalised.
    @test getlogscale(call_message_update_rule(*, :out; m = (A = PointMass(c), in = out))) === 0
end

@testitem "rules:logscale:+ and - by quadrature" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    using HCubature: hcubature

    # a ↦ ∫ m_out(a + b) m_in2(b) db integrates to one, as every rule of `+` and `-` does.
    out, in2 = NormalMeanVariance(1.0, 2.0), NormalMeanVariance(-0.5, 0.7)
    total = first(hcubature(x -> pdf(out, x[1] + x[2]) * pdf(in2, x[2]), (-30.0, -30.0), (30.0, 30.0); rtol = 1.0e-8))
    @test total ≈ 1 rtol = 1.0e-6
    for (node, target, m) in (
            (+, :in1, (out = out, in2 = in2)), (+, :in2, (out = out, in1 = in2)), (+, :out, (in1 = out, in2 = in2)),
            (-, :in1, (out = out, in2 = in2)), (-, :in2, (out = out, in1 = in2)), (-, :out, (in1 = out, in2 = in2)),
            (+, :out, (in1 = Gamma(2.0, 1.0), in2 = Gamma(3.0, 1.0))), (-, :in1, (out = Gamma(2.0, 1.0), in2 = Gamma(3.0, 1.0))),
        )
        @test getlogscale(call_message_update_rule(node, target; m)) === 0
    end
    # A known vector's dot product with a Gaussian is its pushforward.
    @test getlogscale(call_message_update_rule(dot, :out; m = (in1 = PointMass([1.0, 2.0]), in2 = MvNormalMeanCovariance([0.0, 1.0], [1.0 0.0; 0.0 1.0])))) === 0
    @test getlogscale(call_message_update_rule(dot, :out; m = (in1 = MvNormalMeanCovariance([0.0, 1.0], [1.0 0.0; 0.0 1.0]), in2 = PointMass([1.0, 2.0])))) === 0
end
