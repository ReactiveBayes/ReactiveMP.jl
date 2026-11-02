# The engine consults a rule fallback through `call_fallback` and `call_energy_fallback`, with the
# node's algorithm and its rule context: for a message, a marginal and an average energy that no
# rule gives. A three-argument fallback is asked about messages only.

@testmodule ContextFallbackNodes begin
    using MessagePassingRulesBase, BayesBase, ExponentialFamily

    # `out ~ Toy(a)`, a unit-variance Gaussian log-density in `out` with mean `a`, unnormalised: its
    # rules exist, but no marginal rule and no average energy.
    struct Toy{A}
        a::A
    end
    BayesBase.logpdf(d::Toy, x) = -(x - d.a)^2 / 2
    @define_factor_node(node = Toy, type = Stochastic, interfaces = [:out, :a])
    @define_message_update_rule(node = Toy, target = :a, args = (q[:out]::Any,), body = (args) -> NormalMeanVariance(mean(args.q[:out]), 1.0))
    @define_message_update_rule(node = Toy, target = :out, args = (q[:a]::Any,), body = (args) -> NormalMeanVariance(mean(args.q[:a]), 1.0))

    # A fallback that records what the engine gives it, and answers every question.
    struct Recorder
        calls::Vector{Any}
    end
    Recorder() = Recorder(Any[])
    MessagePassingRulesBase.call_fallback(r::Recorder, node, target, algorithm, ctx, args) =
        (push!(r.calls, (; node, target, algorithm, ctx)); PointMass(42.0))
    MessagePassingRulesBase.call_energy_fallback(r::Recorder, node, algorithm, ctx, args) =
        (push!(r.calls, (; node, target = :energy, algorithm, ctx)); 7.0)

    # A three-point rule, exact for polynomials of degree five under a Gaussian.
    struct ThreePoints end
    const Gaussian = ExponentialFamily.UnivariateNormalDistributionsFamily
    BayesBase.supports_expectation(::ThreePoints, ::Type, ::Type{<:Union{Gaussian, PointMass}}) = true
    function BayesBase.prepare_expectation(::ThreePoints, q::Gaussian)
        s = sqrt(3 * var(q))
        return BayesBase.ExpectationPointSet((mean(q) - s, mean(q), mean(q) + s), (1 / 6, 2 / 3, 1 / 6))
    end
    BayesBase.prepare_expectation(::ThreePoints, q::PointMass) = BayesBase.ExpectationPointSet((mean(q),), (1.0,))

    # A fallback that knows its message's log scale.
    struct Scaled end
    MessagePassingRulesBase.call_fallback(::Scaled, node, target, algorithm, ctx, args) = with_logscale(PointMass(42.0), -0.5)
end

@testitem "rulefallback: a message fallback gets the algorithm and the rule context" tags = [:engine] setup = [ContextFallbackNodes] begin
    import ReactiveMP: MessageMapping, EngineDiagnostics, getdata
    import MessagePassingRulesBase: Target, DefaultAlgorithm
    using BayesBase
    N = ContextFallbackNodes

    # `Toy` has no rule towards `out` from a message.
    recorder = N.Recorder()
    instance = N.Toy(0.0)
    mapping = MessageMapping(N.Toy, Target{:out}(), Val((:a,)), nothing, DefaultAlgorithm(), nothing, instance, nothing, EngineDiagnostics(), (scale = 2,), recorder)
    @test getdata(mapping((Message(PointMass(1.0), false, false),), nothing)) === PointMass(42.0)
    call = only(recorder.calls)
    @test call.node === N.Toy && call.target === Target{:out}() && call.algorithm === DefaultAlgorithm()
    @test call.ctx.node === instance && call.ctx.scale == 2 && call.ctx.expectation === nothing
end

@testitem "rulefallback: a fallback may declare its message's log scale" tags = [:engine] setup = [ContextFallbackNodes] begin
    import ReactiveMP: MessageMapping, MarginalMapping, EngineDiagnostics, getdata, compute_marginal
    import MessagePassingRulesBase: Target, ClusterTarget, DefaultAlgorithm
    using BayesBase
    N = ContextFallbackNodes

    mapping(fallback, logscales) = MessageMapping(N.Toy, Target{:out}(), Val((:a,)), nothing, DefaultAlgorithm(), nothing, N.Toy(0.0), nothing, EngineDiagnostics(), nothing, fallback, logscales)
    inputs = (Message(PointMass(1.0), false, false),)
    # The declared log scale where log scales are tracked, and the message alone where they are not.
    message = mapping(N.Scaled(), true)(inputs, nothing)
    @test getdata(message) === PointMass(42.0) && getlogscale(message) === -0.5
    untracked = mapping(N.Scaled(), false)(inputs, nothing)
    @test getdata(untracked) === PointMass(42.0) && untracked.logscale === nothing
    # A fallback that declares none leaves it undefined, as before.
    @test getlogscale(mapping(N.Recorder(), true)(inputs, nothing)).cause === :fallback
    # A joint has no log scale: the declared one is dropped.
    messages = (Message(PointMass(1.0), false, false), Message(PointMass(2.0), false, false))
    joint = MarginalMapping(N.Toy, ClusterTarget((:out, :a)), Val((:out, :a)), nothing, DefaultAlgorithm(), N.Toy(0.0), EngineDiagnostics(), nothing, nothing, N.Scaled())
    @test compute_marginal(joint, messages, nothing) === PointMass(42.0)
end

@testitem "rulefallback: a marginal without a rule" tags = [:engine] setup = [ContextFallbackNodes] begin
    import ReactiveMP: MarginalMapping, EngineDiagnostics, RuleNotFoundError, compute_marginal
    import MessagePassingRulesBase: ClusterTarget, DefaultAlgorithm, NodeFunctionRuleFallback
    using BayesBase
    N = ContextFallbackNodes

    messages = (Message(PointMass(1.0), false, false), Message(PointMass(2.0), false, false))
    mapping(fallback) = MarginalMapping(N.Toy, ClusterTarget((:out, :a)), Val((:out, :a)), nothing, DefaultAlgorithm(), N.Toy(0.0), EngineDiagnostics(), nothing, nothing, fallback)

    # Without a fallback, and with one written for messages only, the error as before.
    @test MarginalMapping(N.Toy, ClusterTarget((:out, :a)), Val((:out, :a)), nothing, DefaultAlgorithm(), N.Toy(0.0)).rulefallback === nothing
    @test_throws RuleNotFoundError compute_marginal(mapping(nothing), messages, nothing)
    @test_throws RuleNotFoundError compute_marginal(mapping((fform, target, args) -> PointMass(0.0)), messages, nothing)
    @test_throws RuleNotFoundError compute_marginal(mapping(NodeFunctionRuleFallback()), messages, nothing)

    # A fallback that answers about marginals gives the joint.
    recorder = N.Recorder()
    @test compute_marginal(mapping(recorder), messages, nothing) === PointMass(42.0)
    @test only(recorder.calls).target === ClusterTarget((:out, :a)) && only(recorder.calls).algorithm === DefaultAlgorithm()
end

@testitem "rulefallback: an average energy without a rule" tags = [:engine] setup = [ContextFallbackNodes] begin
    import ReactiveMP: activate!, factornode, node_activation, average_energy_term, FactorNodeActivationOptions,
        RandomVariableActivationOptions, MessageProductContext, RuleNotFoundError
    import MessagePassingRulesBase: DefaultAlgorithm, NodeFunctionRuleFallback
    using BayesBase, ExponentialFamily
    N = ContextFallbackNodes

    qout, qa = NormalMeanVariance(1.0, 0.5), NormalMeanVariance(0.25, 0.1)
    marginals = (Marginal(qout, false, false), Marginal(qa, false, false))
    entropies = entropy(qout) + entropy(qa)
    function activated(options)
        out, a = randomvar(), randomvar()
        node = factornode(N.Toy, [(:out, out), (:a, a)], ((:out,), (:a,)))
        product = MessageProductContext()
        foreach(v -> activate!(v, RandomVariableActivationOptions(nothing, product, product)), (out, a))
        activate!(node, options)
        return node
    end
    term(node) = average_energy_term(Float64, node_activation(node), node, DefaultAlgorithm(), Val((:out, :a)), marginals)

    # Without a fallback, and with one written for messages only, the error as before.
    @test_throws RuleNotFoundError term(activated(FactorNodeActivationOptions()))
    @test_throws RuleNotFoundError term(activated(FactorNodeActivationOptions(; rulefallback = (fform, target, args) -> PointMass(0.0))))
    # `NodeFunctionRuleFallback` needs an expectation backend in the context.
    @test_throws RuleNotFoundError term(activated(FactorNodeActivationOptions(; rulefallback = NodeFunctionRuleFallback())))

    # A fallback that answers gives the average energy, its node's activation context with it.
    recorder = N.Recorder()
    node = activated(FactorNodeActivationOptions(; rulefallback = recorder, context = (scale = 3,)))
    @test term(node) ≈ 7.0 - entropies
    call = only(recorder.calls)
    @test call.node === N.Toy && call.algorithm === DefaultAlgorithm() && call.ctx.scale == 3

    # `-E_q[log Toy(a)(out)] = ((E[out] - E[a])² + Var[out] + Var[a]) / 2`, through the context's backend.
    node = activated(FactorNodeActivationOptions(; rulefallback = NodeFunctionRuleFallback(), context = (expectation = N.ThreePoints(),)))
    @test term(node) ≈ ((1.0 - 0.25)^2 + 0.5 + 0.1) / 2 - entropies
end

@testitem "rulefallback: a free energy through the average-energy fallback" tags = [:engine] setup = [EngineHarness, ContextFallbackNodes] begin
    # a ~ N(0, 1), y ~ Toy(a) with Toy(a)(y) = exp(-(y - a)² / 2) = √(2π) N(y; a, 1): the posterior
    # is exact after one iteration, and the free energy is -log ∫ N(a; 0, 1) Toy(a)(y) da
    # = -log N(y; 0, 2) - log √(2π).
    using ExponentialFamily, Distributions, StandardMessagePassingRules
    import MessagePassingRulesBase: NodeFunctionRuleFallback
    H, N = EngineHarness, ContextFallbackNodes

    graph = H.Graph()
    a = H.random!(graph)
    y = H.data!(graph)
    H.node!(graph, NormalMeanVariance, [(:out, a), (:μ, H.constant!(graph, 0.0)), (:v, H.constant!(graph, 1.0))])
    interfaces = [(:out, y), (:a, a)]
    H.node!(graph, N.Toy, interfaces; factorisation = H.meanfield_factorisation(interfaces))

    run(context) = H.run(graph; data = [y => 0.8], iterations = 3, posteriors = [:a => a], context, rulefallback = NodeFunctionRuleFallback())
    result = run((expectation = N.ThreePoints(),))
    @test mean(result.posteriors["a"]) ≈ 0.4 && var(result.posteriors["a"]) ≈ 0.5
    @test result.free_energy ≈ fill(-logpdf(Normal(0.0, sqrt(2.0)), 0.8) - log(2π) / 2, 3)
end
