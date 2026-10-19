# What a trace of a whole graph shows: the rule calls, the joints and the free energy's terms.

@testmodule TracingGraph begin
    using ReactiveMP, Rocket, BayesBase, ExponentialFamily, StandardMessagePassingRules
    import ReactiveMP:
        activate!, FactorNodeActivationOptions, RandomVariableActivationOptions, DataVariableActivationOptions,
        MessageProductContext, get_stream_of_marginals

    """
    `x ~ N(0, 10)`, `w ~ N(1, 2)`, `s = x + w`, `y ~ N(s, 1)` with `y` observed, its factor nodes
    activated with `node_callbacks` and its random variables with `variable_callbacks`, as RxInfer
    gives both. Belief propagation: the free energy is minus the log evidence, `y ~ N(1, 13)`.
    """
    function sum_of_gaussians(; node_callbacks = nothing, variable_callbacks = nothing)
        x, w, s, y = randomvar(label = :x), randomvar(label = :w), randomvar(label = :s), datavar(label = :y)
        constants = [constvar(0.0), constvar(10.0), constvar(1.0), constvar(2.0), constvar(1.0)]
        prior_x = factornode(NormalMeanVariance, [(:out, x), (:μ, constants[1]), (:v, constants[2])], ((:out,), (:μ,), (:v,)))
        prior_w = factornode(NormalMeanVariance, [(:out, w), (:μ, constants[3]), (:v, constants[4])], ((:out,), (:μ,), (:v,)))
        sum_node = factornode(+, [(:out, s), ((:in, 1), x), ((:in, 2), w)])
        likelihood = factornode(NormalMeanVariance, [(:out, y), (:μ, s), (:v, constants[5])], ((:out,), (:μ,), (:v,)))
        nodes = (prior_x, prior_w, sum_node, likelihood)
        for variable in (x, w, s)
            activate!(variable, RandomVariableActivationOptions(nothing, MessageProductContext(), MessageProductContext(; callbacks = variable_callbacks)))
        end
        activate!(y, DataVariableActivationOptions())
        foreach(node -> activate!(node, FactorNodeActivationOptions(; callbacks = node_callbacks)), nodes)
        return (; x, w, s, y, constants, nodes, sum_node, variables = [x, w, s, y, constants...])
    end

    export sum_of_gaussians
end

@testitem "engine:tracing: every free-energy term is reported, and the terms sum to the free energy" tags = [:engine] setup = [TracingGraph] begin
    using ReactiveMP, Rocket, BayesBase, Distributions, MessagePassingRulesBase
    import ReactiveMP: event_name, get_stream_of_marginals, degree

    events = Any[]
    record = (event) -> (push!(events, event); nothing)
    names = (
        :before_message_rule_call, :after_message_rule_call, :before_marginal_rule_call, :after_marginal_rule_call,
        :before_factor_bound_free_energy, :after_factor_bound_free_energy, :before_variable_bound_entropy, :after_variable_bound_entropy,
    )
    callbacks = NamedTuple{names}(ntuple(_ -> record, length(names)))
    graph = sum_of_gaussians(; node_callbacks = callbacks, variable_callbacks = callbacks)

    energies = Float64[]
    subscriptions = [
        subscribe!(get_stream_of_marginals(graph.x), (q) -> nothing),
        subscribe!(bethe_free_energy(Float64, graph.nodes, graph.variables), (f) -> push!(energies, f)),
    ]
    new_observation!(graph.y, 3.0)
    foreach(unsubscribe!, subscriptions)

    @test only(energies) ≈ -logpdf(Normal(1.0, sqrt(13.0)), 3.0)

    named(name) = filter(event -> event_name(event) === name, events)
    node_terms = named(:after_factor_bound_free_energy)
    variable_terms = named(:after_variable_bound_entropy)
    @test length(node_terms) == length(graph.nodes)
    @test all(node -> count(event -> event.node === node, node_terms) == 1, graph.nodes)
    @test Set(event.variable for event in variable_terms) == Set([graph.x, graph.w, graph.s])

    # The terms, less one point entropy per connection of a data or constant variable, are the
    # free energy, as `bethe_free_energy` sums them.
    points = sum(degree, [graph.y; graph.constants])
    total = sum(event.result for event in node_terms) + sum(event.result for event in variable_terms) - BayesBase.CountingReal(Float64, points)
    @test float(total) ≈ only(energies)

    # A term counts the infinite entropies of point masses apart from its value.
    counting(x) = convert(BayesBase.CountingReal{Float64}, x)
    same(a, b) = BayesBase.infinities(counting(a)) == BayesBase.infinities(counting(b)) && BayesBase.value(counting(a)) ≈ BayesBase.value(counting(b))
    for event in node_terms
        if event.node === graph.sum_node
            @test event.energy === nothing
            @test event.rule === nothing
            @test same(event.result, -event.entropy)
            @test only(event.marginals) isa Marginal
        else
            @test event.rule isa MessagePassingRulesBase.RuleSpec
            @test event.rule.kind === :average_energy
            @test same(event.result, event.energy - event.entropy)
            @test length(event.marginals) == 3
        end
    end
    for event in variable_terms
        @test event.scaling == degree(event.variable) - 1
        @test same(event.result, event.scaling * event.entropy)
    end

    # The joint the deterministic node's term reads is computed by its marginal rule, inside the
    # term's span.
    before = findfirst(event -> event_name(event) === :before_factor_bound_free_energy && event.node === graph.sum_node, events)
    after = findfirst(event -> event_name(event) === :after_factor_bound_free_energy && event.node === graph.sum_node, events)
    inside = filter(event -> event_name(event) === :after_marginal_rule_call, events[before:after])
    @test length(inside) == 1
    @test only(inside).mapping.factornode === graph.sum_node
    @test only(inside).rule isa MessagePassingRulesBase.RuleSpec
    @test only(inside).rule.kind === :marginal

    # Every message names the rule that gave it.
    @test all(event -> event.rule isa MessagePassingRulesBase.RuleSpec && event.rule.kind === :message, named(:after_message_rule_call))

    # Every "after" event follows its "before" event, paired by span.
    pairs = (
        :after_message_rule_call => :before_message_rule_call,
        :after_marginal_rule_call => :before_marginal_rule_call,
        :after_factor_bound_free_energy => :before_factor_bound_free_energy,
        :after_variable_bound_entropy => :before_variable_bound_entropy,
    )
    for (after_name, before_name) in pairs, (k, event) in enumerate(events)
        event_name(event) === after_name || continue
        opened = findall(other -> event_name(other) === before_name && other.span_id == event.span_id, events)
        @test length(opened) == 1 && only(opened) < k
    end
end

@testitem "engine:tracing: a rule event names its rule, the fallback, or nothing for a missing input" tags = [:engine] begin
    using ReactiveMP, BayesBase, ExponentialFamily, MessagePassingRulesBase, StandardMessagePassingRules
    import ReactiveMP: MessageMapping, marginal_mapping, marginal_from_inputs, node_context, EngineDiagnostics
    import MessagePassingRulesBase: Target, ClusterTarget, DefaultAlgorithm, RuleSpec

    struct Opaque end
    MessagePassingRulesBase.@define_factor_node(node = Opaque, type = Stochastic, interfaces = [:out, :x])

    afters = Any[]
    callbacks = (after_message_rule_call = (event) -> push!(afters, event), after_marginal_rule_call = (event) -> push!(afters, event))

    opaque = factornode(Opaque, [(:out, randomvar()), (:x, randomvar())])
    fallback = (fform, target, args) -> PointMass(1.0)
    towards_out = MessageMapping(Opaque, Target{:out}(), Val((:x,)), nothing, DefaultAlgorithm(), nothing, opaque, callbacks, EngineDiagnostics(), nothing, fallback)
    towards_out((Message(2.0, false, false),), nothing)
    @test last(afters).rule === fallback
    towards_out((Message(missing, false, false),), nothing)
    @test last(afters).rule === nothing

    gaussian = factornode(NormalMeanVariance, [(:out, randomvar()), (:μ, randomvar()), (:v, constvar(1.0))], ((:out, :μ), (:v,)))
    towards_mean = MessageMapping(NormalMeanVariance, Target{:μ}(), Val((:out, :v)), nothing, DefaultAlgorithm(), nothing, gaussian, callbacks)
    towards_mean((Message(NormalMeanVariance(0.0, 1.0), false, false), Message(PointMass(1.0), true, false)), nothing)
    @test last(afters).rule isa RuleSpec
    @test last(afters).rule.node === NormalMeanVariance
    @test last(afters).rule.target === Target{:μ}

    joint = marginal_mapping(NormalMeanVariance, ClusterTarget((:out, :μ)), Val((:out, :μ)), Val((:v,)), DefaultAlgorithm(), gaussian, EngineDiagnostics(), node_context(gaussian), callbacks)
    marginals = (Marginal(PointMass(1.0), true, false),)
    marginal_from_inputs(joint, (Message(NormalMeanVariance(1.0, 2.0), false, false), Message(NormalMeanVariance(0.0, 1.0), false, false)), marginals)
    @test last(afters).rule isa RuleSpec
    @test last(afters).rule.kind === :marginal
    @test last(afters).result !== missing
    marginal_from_inputs(joint, (Message(missing, false, false), Message(NormalMeanVariance(0.0, 1.0), false, false)), marginals)
    @test last(afters).rule === nothing
    @test last(afters).result === missing
end

@testitem "engine:tracing: with no handler listening, the traced paths cost nothing" tags = [:engine, :alloc] setup = [TracingGraph] begin
    using ReactiveMP, Rocket, BayesBase, ExponentialFamily, MessagePassingRulesBase, StandardMessagePassingRules
    import ReactiveMP: MessageMapping, marginal_mapping, marginal_from_inputs, compute_marginal, node_context, EngineDiagnostics,
        FactorBoundFreeEnergy, VariableBoundEntropy, get_stream_of_marginals
    import MessagePassingRulesBase: Target, ClusterTarget, DefaultAlgorithm

    # A handler that listens to no event and generates no span ids: the engine then builds nothing.
    struct Deaf end
    ReactiveMP.listens(::Deaf, ::Type{<:ReactiveMP.Event}) = false
    ReactiveMP.generate_span_id(::Deaf) = nothing

    # Measured through a function of the arguments, which specialises on them, so that the
    # test's own globals are not counted.
    function allocations(f::F, args::Vararg{Any, N}) where {F, N}
        f(args...)
        return @allocated f(args...)
    end

    gaussian = factornode(NormalMeanVariance, [(:out, randomvar()), (:μ, randomvar()), (:v, constvar(1.0))], ((:out, :μ), (:v,)))
    messages = (Message(NormalMeanVariance(0.0, 1.0), false, false), Message(PointMass(1.0), true, false))
    towards_out(callbacks) = MessageMapping(NormalMeanVariance, Target{:out}(), Val((:μ, :v)), nothing, DefaultAlgorithm(), nothing, gaussian, callbacks)
    rule_call(mapping, messages) = mapping(messages, nothing)
    @test @inferred(rule_call(towards_out(nothing), messages)) isa Message
    @test allocations(rule_call, towards_out(Deaf()), messages) == allocations(rule_call, towards_out(nothing), messages)

    # A joint between its events allocates what its rule alone does.
    joint = marginal_mapping(NormalMeanVariance, ClusterTarget((:out, :μ)), Val((:out, :μ)), Val((:v,)), DefaultAlgorithm(), gaussian, EngineDiagnostics(), node_context(gaussian), nothing)
    inputs = (Message(NormalMeanVariance(1.0, 2.0), false, false), Message(NormalMeanVariance(0.0, 1.0), false, false))
    marginals = (Marginal(PointMass(1.0), true, false),)
    traced(joint, inputs, marginals) = marginal_from_inputs(joint, inputs, marginals)
    untraced(joint, inputs, marginals) = compute_marginal(joint, inputs, marginals)
    @test @inferred(traced(joint, inputs, marginals)) == untraced(joint, inputs, marginals)
    @test allocations(traced, joint, inputs, marginals) == allocations(untraced, joint, inputs, marginals)

    # A graph whose free-energy terms are subscribed allocates as much under the deaf handler as
    # under none.
    function observation_bytes(callbacks)
        graph = sum_of_gaussians(; node_callbacks = callbacks, variable_callbacks = callbacks)
        T = BayesBase.CountingReal{Float64}
        subscriptions = Any[subscribe!(get_stream_of_marginals(graph.x), (q) -> nothing)]
        for node in graph.nodes
            push!(subscriptions, subscribe!(score(T, FactorBoundFreeEnergy(), node, nothing, nothing), (v) -> nothing))
        end
        for variable in (graph.x, graph.w, graph.s)
            push!(subscriptions, subscribe!(score(T, VariableBoundEntropy(), variable, nothing), (v) -> nothing))
        end
        new_observation!(graph.y, 1.0)
        new_observation!(graph.y, 2.0)
        bytes = @allocated new_observation!(graph.y, 3.0)
        foreach(unsubscribe!, subscriptions)
        return bytes
    end
    observation_bytes(nothing), observation_bytes(Deaf())
    @test observation_bytes(Deaf()) == observation_bytes(nothing)
end
