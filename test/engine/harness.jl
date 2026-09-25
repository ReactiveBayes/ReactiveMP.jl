# Runs a graph through the engine the way RxInfer would, and records what happened: every rule
# call, every posterior update and the free energy after each iteration.

@testmodule EngineHarness begin
    using ReactiveMP, Rocket, BayesBase, MessagePassingRulesBase
    import ReactiveMP:
        activate!, israndom, isdata, getdata, getannotations, has_annotation, get_annotation, message_mapping_fform,
        FactorNodeActivationOptions, RandomVariableActivationOptions, DataVariableActivationOptions,
        MessageProductContext, get_stream_of_marginals, get_stream_of_predictions, set_initial_marginal!, set_initial_message!
    import MessagePassingRulesBase: Target, IndexedTarget

    # `out := in`, for a graph that needs a variable to receive a point mass.
    struct Copy end
    MessagePassingRulesBase.@define_factor_node(node = Copy, type = Deterministic, interfaces = [:out, :in])
    MessagePassingRulesBase.@define_message_update_rule(node = Copy, target = :out, args = (m[:in]::Any,), body = (args) -> args.m[:in])
    MessagePassingRulesBase.@define_message_update_rule(node = Copy, target = :in, args = (m[:out]::Any,), body = (args) -> args.m[:out])

    """
    A graph in the making: variables and factor nodes in the order they were created, which
    is the order RxInfer creates and activates them in, and the algorithm a node was given.
    """
    struct Graph
        variables::Vector{Any}
        nodes::Vector{Any}
        algorithms::IdDict{Any, Any}
        Graph() = new(Any[], Any[], IdDict{Any, Any}())
    end

    random!(graph::Graph) = (v = randomvar(); push!(graph.variables, v); v)
    data!(graph::Graph) = (v = datavar(); push!(graph.variables, v); v)
    constant!(graph::Graph, value) = (v = constvar(value); push!(graph.variables, v); v)

    # RxInfer's default factorisation: the random interfaces form one cluster, and every data
    # or constant interface is a cluster of its own.
    function bethe_factorisation(interfaces)
        random = Tuple(name for (name, variable) in interfaces if israndom(variable))
        others = Tuple((name,) for (name, variable) in interfaces if !israndom(variable))
        return isempty(random) ? others : (random, others...)
    end

    # RxInfer's `MeanField()`: every interface is a cluster of its own.
    meanfield_factorisation(interfaces) = Tuple((name,) for (name, _) in interfaces)

    # `algorithm` is RxInfer's per-node option; `nothing` is the node's default.
    function node!(graph::Graph, fform, interfaces; factorisation = bethe_factorisation(interfaces), algorithm = nothing, nodefn = nothing)
        node = factornode(fform, interfaces, factorisation; nodefn)
        push!(graph.nodes, node)
        graph.algorithms[node] = algorithm
        return node
    end

    """
    One rule call: the iteration it happened in (0 before the first data), the node's functional
    form and the target, as `":out"` or `"(:in, 1)"`, the result and its log scale: a number, an
    `UndefinedLogScale`, or `nothing` when log scales are not tracked.
    """
    struct RuleCall
        iteration::Int
        node::String
        target::String
        result::Any
        logscale::Any
    end

    """
    What a run produced: `posteriors[name]` is the last marginal of a watched variable (a vector
    for a vector of variables), `history[name]` every marginal it was given, in order, and
    `predictions` the last prediction of each predicted data variable (`nothing` if none came).
    `free_energy` has one value per iteration, `trace` every rule call, and `logscales[name]` the
    last marginal's log scale when a run tracks log scales.
    """
    struct Run
        id::String
        posteriors::Dict{String, Any}
        history::Dict{String, Any}
        predictions::Vector{Any}
        free_energy::Vector{Float64}
        trace::Vector{RuleCall}
        logscales::Dict{String, Any}
    end

    target_text(target::Target{E}) where {E} = ":$E"
    target_text(target::IndexedTarget{E}) where {E} = "(:$E, $(target.index))"

    unwrapall(history::Vector) = map(getdata, history)
    unwrapall(histories::Vector{<:Vector}) = map(unwrapall, histories)
    final(history::Vector) = getdata(last(history))
    final(histories::Vector{<:Vector}) = map(final, histories)

    """
        run(graph; data, iterations, posteriors, id = "", predictions = [], initial_marginals = [], initial_messages = [], logscales = false, free_energy = true)

    Activate `graph` as RxInfer does (variables, each followed by its entry in
    `initial_marginals` and then in `initial_messages` (variable => distribution), the latter set
    on every message out of the variable, as RxInfer's `μ(x)` is; then factor nodes), subscribe to the
    `posteriors` (name => variable or vector of variables), then to the `predictions` of data
    variables, then to the free energy, and feed `data` (variable => value, or vectors of
    both) once per iteration. RxInfer predicts a data variable when its data has a `missing`.
    """
    function run(graph::Graph; data, iterations, posteriors, id = "", predictions = [], initial_marginals = [], initial_messages = [], logscales = false, free_energy = true)
        trace = RuleCall[]
        iteration = Ref(0)
        callbacks = (
            after_message_rule_call = (event) -> begin
                node = string(nameof(message_mapping_fform(event.mapping)))
                push!(trace, RuleCall(iteration[], node, target_text(event.mapping.target), event.result, event.logscale))
                nothing
            end,
        )

        product = MessageProductContext()
        for variable in graph.variables
            if israndom(variable)
                activate!(variable, RandomVariableActivationOptions(nothing, product, product))
            elseif isdata(variable)
                activate!(variable, DataVariableActivationOptions(true, false, nothing, nothing))
            end
            for (initialised, marginal) in initial_marginals
                initialised === variable && set_initial_marginal!(variable, marginal)
            end
            for (initialised, message) in initial_messages
                initialised === variable && set_initial_message!(variable, message)
            end
        end
        for node in graph.nodes
            activate!(node, FactorNodeActivationOptions(; algorithm = graph.algorithms[node], callbacks, logscales))
        end

        histories = Dict{String, Any}()
        subscriptions = Any[]
        for (name, variables) in posteriors
            watched = variables isa AbstractVector ? variables : [variables]
            records = [Any[] for _ in watched]
            for (record, variable) in zip(records, watched)
                push!(subscriptions, subscribe!(get_stream_of_marginals(variable), (q) -> push!(record, q)))
            end
            histories[string(name)] = variables isa AbstractVector ? records : only(records)
        end
        predicted = Any[nothing for _ in predictions]
        for (k, variable) in enumerate(predictions)
            push!(subscriptions, subscribe!(get_stream_of_predictions(variable), (p) -> (predicted[k] = getdata(p))))
        end
        energies = Float64[]
        if free_energy
            push!(subscriptions, subscribe!(bethe_free_energy(Float64, graph.nodes, graph.variables; algorithm = (node) -> graph.algorithms[node]), (f) -> push!(energies, f)))
        end

        for it in 1:iterations
            iteration[] = it
            for (variable, value) in data
                if variable isa AbstractVector
                    foreach(new_observation!, variable, value)
                else
                    new_observation!(variable, value)
                end
            end
        end
        foreach(unsubscribe!, subscriptions)

        results = Dict{String, Any}(name => final(history) for (name, history) in histories)
        unwrapped = Dict{String, Any}(name => unwrapall(history) for (name, history) in histories)
        final_logscales = Dict{String, Any}()
        if logscales
            for (name, history) in histories
                history isa Vector{<:Vector} && continue
                final_logscales[name] = getlogscale(last(history))
            end
        end
        return Run(id, results, unwrapped, predicted, energies, trace, final_logscales)
    end
end
