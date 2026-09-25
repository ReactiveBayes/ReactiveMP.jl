@testitem "engine:retained-values" tags = [:engine] setup = [EngineHarness] begin
    # Once a message is materialised, its value, annotations and log scale never change underneath
    # whoever holds it, however many updates follow. A deferred message is computed when first
    # read, so the guarantee starts there.
    using ExponentialFamily, StandardMessagePassingRules
    import ReactiveMP:
        activate!, as_message, getdata, getannotations, get_annotation, getinterfaces,
        get_stream_of_outbound_messages, get_stream_of_marginals, InputArgumentsAnnotations,
        FactorNodeActivationOptions, RandomVariableActivationOptions, DataVariableActivationOptions,
        MessageProductContext, AnnotationDict
    using Rocket
    H = EngineHarness

    annotations = (InputArgumentsAnnotations(),)
    graph = H.Graph()
    y = H.data!(graph)
    x = H.random!(graph)
    prior = H.node!(graph, NormalMeanVariance, [(:out, x), (:μ, H.constant!(graph, 0.0)), (:v, H.constant!(graph, 10.0))])
    likelihood = H.node!(graph, NormalMeanVariance, [(:out, y), (:μ, x), (:v, H.constant!(graph, 1.0))])

    product = MessageProductContext(; annotations)
    activate!(x, RandomVariableActivationOptions(nothing, product, product))
    activate!(y, DataVariableActivationOptions(false, false, nothing, nothing))
    foreach(node -> activate!(node, FactorNodeActivationOptions(; annotations, logscales = true)), graph.nodes)

    towards_x = get_stream_of_outbound_messages(getinterfaces(likelihood)[2])
    held_messages = []
    held_marginals = []
    subscriptions = [
        subscribe!(towards_x, (message) -> push!(held_messages, as_message(message))),
        subscribe!(get_stream_of_marginals(x), (marginal) -> push!(held_marginals, marginal)),
    ]

    snapshot(held) = map(h -> (getdata(h), AnnotationDict(getannotations(h)), getlogscale(h)), held)

    new_observation!(y, 1.0)
    first_messages, first_marginals = snapshot(held_messages), snapshot(held_marginals)
    @test length(first_messages) == 1 && length(first_marginals) == 1
    @test getlogscale(only(held_messages)) isa Real && getlogscale(only(held_marginals)) isa Real

    for value in (2.0, -1.0, 0.5)
        new_observation!(y, value)
    end
    foreach(unsubscribe!, subscriptions)

    @test length(held_messages) == 4 && length(held_marginals) == 4
    # The first values held are exactly what they were
    @test snapshot(held_messages[1:1]) == first_messages
    @test snapshot(held_marginals[1:1]) == first_marginals
    # and every later update is a new object, not the old one changed
    @test allunique(map(objectid, held_messages))
    @test allunique(map(objectid, held_marginals))
    @test getdata(held_messages[1]) != getdata(held_messages[2])
end
