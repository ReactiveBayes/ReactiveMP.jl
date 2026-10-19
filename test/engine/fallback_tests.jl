# A rule fallback, the activation option `rulefallback`, gives a message where no rule matches;
# it is consulted only then, so a rule that exists always wins, and an error inside it propagates.

@testmodule RuleFallbackNodes begin
    using MessagePassingRulesBase, BayesBase

    # `out ~ Toy(a)`, a Gaussian log-density in `out` with mean `a`, with a rule towards `a` only.
    struct Toy{A}
        a::A
    end
    BayesBase.logpdf(d::Toy, x) = -(x - d.a)^2 / 2
    @define_factor_node(node = Toy, type = Stochastic, interfaces = [:out, :a])
    @define_message_update_rule(node = Toy, target = :a, args = (m[:out]::PointMass,), body = (args) -> PointMass(mean(args.m[:out])))

    # A node whose one rule fails.
    struct Broken{A}
        a::A
    end
    BayesBase.logpdf(d::Broken, x) = -(x - d.a)^2 / 2
    @define_factor_node(node = Broken, type = Stochastic, interfaces = [:out, :a])
    @define_message_update_rule(node = Broken, target = :out, args = (m[:a]::PointMass,), body = (args) -> error("the rule itself failed"))
end

@testitem "rulefallback:mapping" tags = [:engine] setup = [RuleFallbackNodes] begin
    import ReactiveMP: MessageMapping, EngineDiagnostics, FactorNodeActivationOptions, RuleNotFoundError, getdata
    import MessagePassingRulesBase: Target, DefaultAlgorithm, NodeFunctionLogPdf, NodeFunctionRuleFallback
    using BayesBase
    N = RuleFallbackNodes
    fallback = NodeFunctionRuleFallback()

    @test FactorNodeActivationOptions().rulefallback === nothing
    @test FactorNodeActivationOptions(; rulefallback = fallback).rulefallback === fallback

    mapping(node, target, rulefallback) = MessageMapping(node, Target{target}(), Val((:a,)), nothing, DefaultAlgorithm(), nothing, node(0.0), nothing, EngineDiagnostics(), nothing, rulefallback)

    # No rule towards `out`: without a fallback, an error; with one, the node's log-density.
    @test_throws RuleNotFoundError mapping(N.Toy, :out, nothing)((Message(PointMass(1.0), false, false),), nothing)
    message = getdata(mapping(N.Toy, :out, fallback)((Message(PointMass(1.0), false, false),), nothing))
    @test message isa NodeFunctionLogPdf && logpdf(message, 3.0) == logpdf(N.Toy(1.0), 3.0)

    # A rule that exists wins, and an error inside a rule propagates, never falls back.
    towards_a = MessageMapping(N.Toy, Target{:a}(), Val((:out,)), nothing, DefaultAlgorithm(), nothing, N.Toy(0.0), nothing, EngineDiagnostics(), nothing, fallback)
    @test getdata(towards_a((Message(PointMass(2.0), false, false),), nothing)) === PointMass(2.0)
    @test_throws "the rule itself failed" mapping(N.Broken, :out, fallback)((Message(PointMass(1.0), false, false),), nothing)
end

@testitem "rulefallback:through activation" tags = [:engine] setup = [RuleFallbackNodes] begin
    import ReactiveMP: activate!, factornode, get_stream_of_marginals,
        FactorNodeActivationOptions, RandomVariableActivationOptions, MessageProductContext, getdata
    import MessagePassingRulesBase: NodeFunctionLogPdf, NodeFunctionRuleFallback
    using Rocket, BayesBase
    N = RuleFallbackNodes

    # With `a` a constant, the message towards `out` has no rule and comes from the fallback.
    out = randomvar()
    node = factornode(N.Toy, [(:out, out), (:a, constvar(1.5))])
    product = MessageProductContext()
    activate!(out, RandomVariableActivationOptions(nothing, product, product))
    activate!(node, FactorNodeActivationOptions(; rulefallback = NodeFunctionRuleFallback()))
    received = []
    # `out` has no other factor, so its marginal is the node's message.
    subscription = subscribe!(get_stream_of_marginals(out), (q) -> push!(received, getdata(q)))
    unsubscribe!(subscription)
    @test !isempty(received) && last(received) isa NodeFunctionLogPdf && logpdf(last(received), 2.0) == logpdf(N.Toy(1.5), 2.0)
end
