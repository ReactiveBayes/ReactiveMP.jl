# Runs a graph through the engine the way RxInfer would, and records what happened as an
# `EngineTrajectory`, to be compared with the trajectories v6 recorded through RxInfer.

@testmodule EngineHarness begin
    using ReactiveMP, Rocket, BayesBase, MessagePassingRulesBase
    using MessagePassingRulesTestUtils: EngineTrajectory, RuleCallRecord, load_engine_fixture
    import ReactiveMP:
        activate!, israndom, isdata, getdata, getannotations, has_annotation, get_annotation, message_mapping_fform,
        FactorNodeActivationOptions, RandomVariableActivationOptions, DataVariableActivationOptions,
        MessageProductContext, get_stream_of_marginals, get_stream_of_predictions, set_initial_marginal!, set_initial_message!
    import MessagePassingRulesBase: Target, IndexedTarget

    const FIXTURES = joinpath(pkgdir(ReactiveMP), "compat", "v6-comparison", "fixtures", "engine")

    # `out := in`, for a graph that needs a variable to receive a point mass.
    struct Copy end
    MessagePassingRulesBase.@define_factor_node(node = Copy, type = Deterministic, interfaces = [:out, :in])
    MessagePassingRulesBase.@define_message_update_rule(node = Copy, target = :out, args = (m[:in]::Any,), body = (args) -> args.m[:in])
    MessagePassingRulesBase.@define_message_update_rule(node = Copy, target = :in, args = (m[:out]::Any,), body = (args) -> args.m[:out])

    fixture(id) = last(load_engine_fixture(joinpath(FIXTURES, "$id.toml")))

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

    target_text(target::Target{E}) where {E} = ":$E"
    target_text(target::IndexedTarget{E}) where {E} = "(:$E, $(target.index))"

    logscale_of(ann) = has_annotation(ann, :logscale) ? Float64(get_annotation(ann, :logscale)) : nothing

    unwrap(q) = getdata(q)
    final(history::Vector) = unwrap(last(history))
    final(histories::Vector{<:Vector}) = map(final, histories)

    """
        run(graph; id, data, iterations, posteriors, predictions = [], initial_marginals = [], initial_messages = [], annotations = nothing, free_energy = true)

    Activate `graph` as RxInfer does (variables, each followed by its entry in
    `initial_marginals` and then in `initial_messages` (variable => distribution), the latter set
    on every message out of the variable, as RxInfer's `μ(x)` is; then factor nodes), subscribe to the
    `posteriors` (name => variable or vector of variables), then to the `predictions` of data
    variables, then to the free energy, and feed `data` (variable => value, or vectors of
    both) once per iteration. RxInfer predicts a data variable when its data has a `missing`.
    """
    function run(graph::Graph; id, data, iterations, posteriors, predictions = [], initial_marginals = [], initial_messages = [], annotations = nothing, free_energy = true)
        trace = RuleCallRecord[]
        iteration = Ref(0)
        callbacks = (
            after_message_rule_call = (event) -> begin
                node = string(nameof(message_mapping_fform(event.mapping)))
                push!(trace, RuleCallRecord(iteration[], node, target_text(event.mapping.target), event.result, logscale_of(event.annotations)))
                nothing
            end,
        )

        product = MessageProductContext(; annotations)
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
            activate!(node, FactorNodeActivationOptions(; algorithm = graph.algorithms[node], annotations, callbacks))
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
        for variable in predictions
            push!(subscriptions, subscribe!(get_stream_of_predictions(variable), (_) -> nothing))
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
        if annotations !== nothing
            for (name, history) in histories
                history isa Vector{<:Vector} && continue
                results["logscale($name)"] = logscale_of(getannotations(last(history)))
            end
        end
        return EngineTrajectory(id; free_energy = energies, posteriors = results, trace)
    end
end
