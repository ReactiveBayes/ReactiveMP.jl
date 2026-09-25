@testmodule FallbackNodes begin
    using MessagePassingRulesBase, BayesBase

    # A node defined only by its log-density: `out ~ Toy(a, b)`, a Gaussian in `out` with mean `a`
    # and variance `b`, with no rules.
    struct Toy{A, B}
        a::A
        b::B
    end
    BayesBase.logpdf(d::Toy, x) = -(x - d.a)^2 / (2 * d.b)
    @define_factor_node(node = Toy, type = Stochastic, interfaces = [:out, :a, :b])

    struct Pair2 end
    @define_factor_node(node = Pair2, type = Deterministic, interfaces = [:out, :in])

    struct Grouped end
    @define_factor_node(node = Grouped, type = Stochastic, interfaces = [:out, :x...])
end

@testitem "fallback:node function" tags = [:base] setup = [FallbackNodes] begin
    using MessagePassingRulesBase, BayesBase
    using MessagePassingRulesBase: RuleArgs, Target, IndexedTarget, NodeFunctionLogPdf
    N = FallbackNodes

    # The node's log-density in the target edge, the other inputs, messages or marginals,
    # collapsed to points by `extract`, their mean by default.
    fallback = NodeFunctionRuleFallback()
    message = fallback(N.Toy, Target(:out), RuleArgs(m = (a = PointMass(1.0),), q = (b = PointMass(2.0),)))
    @test message isa NodeFunctionLogPdf
    @test logpdf(message, 3.0) == logpdf(N.Toy(1.0, 2.0), 3.0) && insupport(message, 3.0)
    towards_a = fallback(N.Toy, Target(:a), RuleArgs(m = (out = PointMass(3.0), b = PointMass(2.0))))
    @test logpdf(towards_a, 1.0) == logpdf(N.Toy(1.0, 2.0), 3.0)
    # Another point.
    lower = NodeFunctionRuleFallback(first)
    @test logpdf(lower(N.Toy, Target(:out), RuleArgs(q = (a = (0.5, 9.0), b = (4.0, 9.0)))), 1.0) == logpdf(N.Toy(0.5, 4.0), 1.0)

    # None where the node function does not apply: a deterministic node, a group member, a joint.
    @test fallback(N.Pair2, Target(:out), RuleArgs(m = (in = PointMass(1.0),))) === nothing
    @test fallback(N.Grouped, IndexedTarget(:x, 1), RuleArgs(m = (out = PointMass(1.0),))) === nothing
    @test fallback(N.Toy, Target(:b), RuleArgs(q = MessagePassingRulesBase.Marginals(NamedTuple(), Val(((:out, :a),)), (PointMass([1.0, 2.0]),)))) === nothing
end
