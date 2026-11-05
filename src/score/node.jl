export FactorBoundFreeEnergy

"""
    FactorBoundFreeEnergy()

Selects a factor node's contribution to the Bethe free energy in [`score`](@ref). A stochastic
node's is its average energy under its local marginals, less the sum of their entropies; a
deterministic node's is minus the entropy of its second cluster: a single input's marginal, or the
joint over its inputs. Under the engine's default dependency scheme its marginal rule computes that
joint from the latest messages on all of its interfaces once none of them is an initial message;
a node whose algorithm declares its dependencies gives the joint its rules read, the cluster's.

The average energy is the one the node's rule package declares for its clusters under the
node's algorithm, found with
[`find_average_energy`](@extref MessagePassingRulesBase.find_average_energy). It runs as the
node's rules do, with the rule context and the diagnostics the node was activated with (see
[`ReactiveMP.FactorNodeActivationOptions`](@ref)), read when the stream is built, or, for a node
not activated by then, when a value is computed; a node not activated at all runs it with the
engine's context, [`ReactiveMP.node_context`](@ref)`(node)`, and no audits.

Each value is reported to the node's callbacks as
[`ReactiveMP.BeforeFactorBoundFreeEnergyEvent`](@ref) and
[`ReactiveMP.AfterFactorBoundFreeEnergyEvent`](@ref), the average energy and the entropies
included; a node not activated reports none.

`FactorBoundFreeEnergy(cavity = true)` is the term under cavity propagation, where the random
variables have a site policy (see [`ReactiveMP.compute_site`](@ref)): a node with no rules under
its algorithm, whose messages its rule fallback sends, takes the fallback's energy from its inbound
messages, the cavities, and its variables' marginals, both given to `call_energy_fallback` as the
rule arguments' `m` and `q` by interface: expectation propagation's term, less the point masses'
entropies. Its stream emits once each of those inputs has updated, once per iteration, since the
sites' variables send their messages and marginals once each. Every other node's term is
Bethe's, as above.
"""
struct FactorBoundFreeEnergy{C} end

FactorBoundFreeEnergy(; cavity::Bool = false) = FactorBoundFreeEnergy{cavity}()

function score(
        ::Type{T},
        objective::FactorBoundFreeEnergy,
        node::AbstractFactorNode,
        algorithm,
        stream_postprocessors,
    ) where {T <: CountingReal}
    fform = functionalform(node)
    return score(
        T,
        objective,
        sdtype(node),
        node,
        something(algorithm, default_algorithm(fform)),
        stream_postprocessors,
    )
end

## Deterministic mapping

# Minus the entropy of the joint over the inputs, the node's second cluster. Under the default
# scheme it is computed by the node's marginal rule from the latest messages on all of its
# interfaces once none of them is an initial message, and again whenever they all have been
# renewed. Not from the cluster's own stream: that one updates as soon as some inputs are renewed,
# so in a graph with loops it can hold a joint computed from an initial message, which no later
# update replaces until every input is renewed, and the free energy of the iterations before
# convergence would not be that of one set of messages. A node whose algorithm declares its
# dependencies has rules that read the cluster, which may be computed by an impure rule, such as
# Delta's under `CVIProjection`: its term is that joint's, from the cluster's stream.
function score(
        ::Type{T},
        ::FactorBoundFreeEnergy,
        ::Deterministic,
        node::AbstractFactorNode,
        algorithm,
        stream_postprocessors,
    ) where {T <: CountingReal}
    joint = last(get_node_local_marginals(getlocalclusters(node)))
    fform = functionalform(node)
    declared = MessagePassingRulesBase.dependencies_spec(fform, algorithm) !== nothing
    # A single input is its variable's own marginal, which is never behind its messages.
    (declared || !isjoint(joint)) && return postprocess_stream_of_scores(
        stream_postprocessors,
        get_stream_of_marginals(joint) |> skip_initial() |> map(T, term_function(joint_entropy_term, T, node_activation(node), node)),
    )
    interfaces = Tuple(getinterfaces(node))
    names = input_names(map(interface -> input_label(node, interface), interfaces))
    messages = combineLatest(map(interface -> get_stream_of_inbound_messages(interface) |> skip_initial(), interfaces), PushNew())
    entropy = term_function(inputs_entropy_term, T, node_activation(node), node, MessagePassingRulesBase.ClusterTarget(name(joint)), names, algorithm)
    stream_of_scores = with_statics(node, messages) |> map(T, entropy)
    return postprocess_stream_of_scores(stream_postprocessors, stream_of_scores)
end

# The function from what a node's term is computed from to the term, `term(T, activation, node,
# parameters..., input)`. An activated node's activation is read here, once, so that its rule
# context, diagnostics and callbacks are concrete in the function, which then calls `term`
# statically; a node not activated yet reads it when each value is computed.
term_function(term::F, ::Type{T}, activation::NodeActivation, node, parameters...) where {F, T} =
    (input) -> term(T, activation, node, parameters..., input)
term_function(term::F, ::Type{T}, ::Nothing, node, parameters...) where {F, T} =
    (input) -> term(T, node_activation(node), node, parameters..., input)

# A deterministic node's term, `-H[q]`, from the joint of its cluster's stream, between the
# free-energy events.
function joint_entropy_term(::Type{T}, activation, node, marginal) where {T}
    callbacks = activation_callbacks(activation)
    span_id = generate_span_id(callbacks)
    @invoke_callback(callbacks, BeforeFactorBoundFreeEnergyEvent(node, span_id))
    entropy = score(DifferentialEntropy(), marginal)
    result = convert(T, -entropy)
    @invoke_callback(callbacks, AfterFactorBoundFreeEnergyEvent(node, (marginal,), nothing, entropy, result, nothing, span_id))
    return result
end

# A deterministic node's term, `-H[q]`, from the joint over its inputs that its marginal rule
# computes from the latest messages, between the free-energy events; the marginal rule call's
# events fall inside them.
function inputs_entropy_term(::Type{T}, activation, node, target, names, algorithm, messages) where {T}
    ctx, diagnostics, callbacks = activation_services(activation, node)
    span_id = generate_span_id(callbacks)
    @invoke_callback(callbacks, BeforeFactorBoundFreeEnergyEvent(node, span_id))
    mapping = marginal_mapping(functionalform(node), target, names, nothing, algorithm, node, diagnostics, ctx, callbacks, activation_rulefallback(activation))
    joint = Marginal(marginal_from_inputs(mapping, messages, nothing), false, false)
    entropy = score(DifferentialEntropy(), joint)
    result = convert(T, -entropy)
    @invoke_callback(callbacks, AfterFactorBoundFreeEnergyEvent(node, (joint,), nothing, entropy, result, nothing, span_id))
    return result
end

## Stochastic mapping

function score(
        ::Type{T},
        ::FactorBoundFreeEnergy,
        ::Stochastic,
        node::AbstractFactorNode,
        algorithm,
        stream_postprocessors,
    ) where {T <: CountingReal}
    fnstream = (localmarginal) -> get_stream_of_marginals(localmarginal) |> skip_initial()

    clusters = getlocalclusters(node)
    localmarginals = get_node_local_marginals(clusters)
    stream = combineLatest(map(fnstream, localmarginals), PushNew())

    marginals_names = input_names(map(i -> cluster_label(node, clusters, i), Tuple(eachindex(localmarginals))))
    mapping = term_function(average_energy_term, T, node_activation(node), node, algorithm, marginals_names)

    stream_of_scores = stream |> map(T, mapping)
    stream_of_scores = postprocess_stream_of_scores(stream_postprocessors, stream_of_scores)

    return stream_of_scores
end

# A stochastic node's term, its average energy less its clusters' entropies, between the
# free-energy events.
function average_energy_term(::Type{T}, activation, node, algorithm, marginals_names, marginals) where {T}
    # The node's type from the node, where its type parameter holds it: a type kept in a closure or
    # a tuple is a `DataType`, which would resolve the average energy at run time on every value.
    fform = functionalform(node)
    ctx, diagnostics, callbacks = activation_services(activation, node)
    span_id = generate_span_id(callbacks)
    @invoke_callback(callbacks, BeforeFactorBoundFreeEnergyEvent(node, span_id))
    args = rule_arguments(nothing, nothing, marginals_names, marginals)
    found = MessagePassingRulesBase.find_average_energy(fform, algorithm, args)
    fallback = activation_rulefallback(activation)
    # `found`'s type is known, so only one branch is compiled.
    outcome = nothing
    if found isa MessagePassingRulesBase.RuleNotFound && !isnothing(fallback)
        spec = fallback
        average_energy, outcome = fallback_energy(fallback, found, fform, algorithm, ctx, args)
    else
        spec = audit_rule(diagnostics, resolve_rule(found))
        MessagePassingRulesBase.check_services(spec, ctx)
        ann = rule_annotations(nothing, nothing, marginals_names, marginals, MessagePassingRulesBase.NoAnnotations())
        average_energy = MessagePassingRulesBase.execute_rule(spec, nothing, MessagePassingRulesBase.rule_algorithm(spec, algorithm), ctx, args, ann, nothing)
    end
    clusters_entropy = mapreduce(marginal -> score(DifferentialEntropy(), marginal), +, marginals)
    result = convert(T, average_energy - clusters_entropy)
    @invoke_callback(callbacks, AfterFactorBoundFreeEnergyEvent(node, marginals, average_energy, clusters_entropy, result, spec, outcome, span_id))
    return result
end

## Under cavity propagation

# A node with no rules of its own under its algorithm, whose messages the rule fallback sends: its
# term is the fallback's from the cavities; every other node's is Bethe's, as without sites.
score(::Type{T}, ::FactorBoundFreeEnergy{true}, type::Stochastic, node::AbstractFactorNode, algorithm, stream_postprocessors) where {T <: CountingReal} =
    cavity_score(T, type, node, algorithm, stream_postprocessors)
score(::Type{T}, ::FactorBoundFreeEnergy{true}, type::Deterministic, node::AbstractFactorNode, algorithm, stream_postprocessors) where {T <: CountingReal} =
    cavity_score(T, type, node, algorithm, stream_postprocessors)

# Whether a node type has rules under an algorithm, once per pair: listing the rules reads every
# loaded module's registry
const HAS_RULES_CACHE = Dict{Tuple{Any, Any}, Bool}()
const HAS_RULES_LOCK = ReentrantLock()

function has_rules(fform, algorithm)
    key = (fform, algorithm)
    cached = @lock HAS_RULES_LOCK get(HAS_RULES_CACHE, key, nothing)
    isnothing(cached) || return cached
    found = !isempty(MessagePassingRulesBase.list_rules(fform; algorithm))
    @lock HAS_RULES_LOCK HAS_RULES_CACHE[key] = found
    return found
end

function cavity_score(::Type{T}, type, node, algorithm, stream_postprocessors) where {T}
    has_rules(functionalform(node), algorithm) &&
        return score(T, FactorBoundFreeEnergy(), type, node, algorithm, stream_postprocessors)
    interfaces = Tuple(getinterfaces(node))
    names = input_names(map(interface -> input_label(node, interface), interfaces))
    messages = map(interface -> get_stream_of_inbound_messages(interface) |> skip_initial(), interfaces)
    marginals = map(interface -> get_stream_of_marginals(getvariable(interface)) |> skip_initial(), interfaces)
    points = Tuple(i for (i, interface) in enumerate(interfaces) if !israndom(getvariable(interface)))
    mapping = term_function(cavity_energy_term, T, node_activation(node), node, algorithm, names, Val(length(interfaces)), points)
    return postprocess_stream_of_scores(stream_postprocessors, combineLatest((messages..., marginals...), PushNew()) |> map(T, mapping))
end

# The fallback's term from the inbound messages, the cavities, and the variables' marginals, both
# by interface, between the free-energy events. Less the entropies of the data's and the
# constants' point masses, `points`, as Bethe's term takes them off its clusters', which the
# point entropies of the free energy cancel.
function cavity_energy_term(::Type{T}, activation, node, algorithm, names, ::Val{N}, points, values) where {T, N}
    messages, marginals = values[1:N], values[(N + 1):end]
    fform = functionalform(node)
    ctx, _, callbacks = activation_services(activation, node)
    fallback = activation_rulefallback(activation)
    span_id = generate_span_id(callbacks)
    @invoke_callback(callbacks, BeforeFactorBoundFreeEnergyEvent(node, span_id))
    args = rule_arguments(names, messages, names, marginals)
    notfound = MessagePassingRulesBase.RuleNotFound(:average_energy, fform, nothing, algorithm, typeof(args))
    isnothing(fallback) && throw(RuleNotFoundError(notfound))
    energy, outcome = fallback_energy(fallback, notfound, fform, algorithm, ctx, args)
    entropy = isempty(points) ? 0.0 : sum(i -> score(DifferentialEntropy(), marginals[i]), points)
    result = convert(T, energy - entropy)
    @invoke_callback(callbacks, AfterFactorBoundFreeEnergyEvent(node, marginals, energy, entropy, result, fallback, outcome, span_id))
    return result
end
