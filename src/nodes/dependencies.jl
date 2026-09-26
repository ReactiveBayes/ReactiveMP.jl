# What each outbound message is computed from, and the streams that deliver it.

# `labels` name each input for the rule (see `input_names`); without them an input is known by
# its `name`, which is enough when no group member is among them.
collect_latest_messages(interfaces) = collect_latest_messages(map(name, Tuple(interfaces)), interfaces)
collect_latest_messages(labels, interfaces) =
    collect_latest_updates(labels, map(get_stream_of_inbound_messages, Tuple(interfaces)), nothing)
collect_latest_marginals(sources) = collect_latest_marginals(map(name, Tuple(sources)), sources)
collect_latest_marginals(labels, sources) =
    collect_latest_updates(labels, map(get_stream_of_marginals, Tuple(sources)), reset_vstatus_of_sources)

# Inputs are subscribed in the order they are listed, which for declared dependencies is the
# declaration's: in variational message passing that order is the update schedule.
function collect_latest_updates(labels, streams::Tuple, callback::C) where {C}
    isempty(streams) && return isempty(labels) ? (nothing, of(nothing)) : (input_names(labels), of(()))
    return (input_names(labels), combineLatestUpdates(streams, PushNew(), typeof(streams), identity, callback))
end

# Mirrors `reset_vstatus` from `clusters.jl`/`random.jl`: without this, a sibling that only ever
# emitted a provisional (`is_initial`) value permanently retires `PushNew()`'s mutual-refresh
# requirement for the rest of the group once it settles on its real value.
# The reset is applied to marginal dependencies only. Applying it to message dependencies as well
# alters the message-update schedule in models that are not affected by the deadlock (an outbound
# message would be recomputed as soon as a single dependency refreshes while the others are still
# `is_initial`), which changes free-energy trajectories and breaks strict FE-monotonicity
# guarantees downstream. Marginal dependencies alone are sufficient to resolve the deadlock.
function reset_vstatus_of_sources(wrapper, sources)
    values = map(getrecent, sources)
    return if is_initial(values)
        Rocket.fill_vstatus!(wrapper, true)
    end
end

"""
    ReactiveMP.input_label(factornode, interface)

What a rule knows an input on `interface` by: its name, or a [`ReactiveMP.GroupMember`](@ref)
for a member of a group.
"""
input_label(factornode, interface::NodeInterface) = name(interface)
input_label(factornode, interface::IndexedNodeInterface) =
    GroupMember(name(interface), index(interface), group_length(factornode, name(interface)))

group_length(factornode, group::Symbol) = count(i -> i isa IndexedNodeInterface && name(i) === group, getinterfaces(factornode))
group_members(factornode, group::Symbol) = filter(i -> i isa IndexedNodeInterface && name(i) === group, getinterfaces(factornode))

# A cluster's label: its interface's for a single interface, the member tuple for a joint.
function cluster_label(factornode, clusters, cindex)
    marginal = get_node_local_marginals(clusters)[cindex]
    isjoint(marginal) && return name(marginal)
    return input_label(factornode, getinterface(factornode, only(getfactorization(clusters, cindex))))
end

"""
    ReactiveMP.default_dependencies(factornode, iindex::Int) -> ((labels, interfaces), (labels, sources))

The engine's default scheme, a regular variational message passing scheme driven by the
factorisation: the message out of interface `iindex` is computed from the inbound messages on
the other interfaces of its cluster, and the marginals of every other cluster. For a
deterministic node it is the inbound messages on every other interface, and no marginal.

Returns the labelled message dependencies and marginal dependencies, each as
`(labels, sources)`: labels from [`ReactiveMP.input_label`](@ref), or a cluster's key, and the
interfaces or local marginals they come from.
"""
function default_dependencies(factornode, iindex)
    # A deterministic node's clusters only say what free energy counts; belief propagation
    # through it reads the messages on every other interface.
    if isdeterministic(sdtype(factornode))
        others = Tuple(i for (j, i) in enumerate(getinterfaces(factornode)) if j != iindex)
        return ((map(i -> input_label(factornode, i), others), others), ((), ()))
    end
    clusters = getlocalclusters(factornode)
    cindex = clusterindex(clusters, iindex)
    cluster = getfactorization(clusters, cindex)
    message_dependencies = map(i -> getinterface(factornode, i), filter(i -> i !== iindex, cluster))
    others = Tuple(i for i in eachindex(get_node_local_marginals(clusters)) if i != cindex)
    return (
        (map(i -> input_label(factornode, i), message_dependencies), message_dependencies),
        (map(i -> cluster_label(factornode, clusters, i), others), map(i -> get_node_local_marginals(clusters)[i], others)),
    )
end

"""
    ReactiveMP.declared_dependencies(factornode, spec::DependenciesSpec, interface) -> ((labels, interfaces), (labels, sources))

What the message out of `interface` is computed from under a declared
[`DependenciesSpec`](@extref MessagePassingRulesBase.DependenciesSpec), in the order the
declaration lists them: `m[:x]` is an interface's inbound message, `q[:x]` its variable's
marginal, and `q[:a, :b]` the local marginal of that cluster. A group dependency selects members
relative to the target's own index. A target declared with `default` is
[`ReactiveMP.extended_default_dependencies`](@ref).

# Throws

- `ArgumentError` when the declaration has no dependencies for the target, asks for the message
  of a cluster, or names a cluster the factorisation does not have.
"""
function declared_dependencies(factornode, spec::MessagePassingRulesBase.DependenciesSpec, interface)
    target = rule_target(interface)
    inputs = MessagePassingRulesBase.target_dependencies(spec, target)
    inputs === nothing && throw(
        ArgumentError("`$(functionalform(factornode))` declares no dependencies for the target `$(repr(interface_key(interface)))` under $(nameof(spec.algorithm))"),
    )
    MessagePassingRulesBase.extends_default_scheme(spec, target) && return extended_default_dependencies(factornode, interface, inputs)
    messages, marginals = (Any[], Any[]), (Any[], Any[])
    for input in inputs
        selected = selected_interfaces(factornode, input, interface)
        collection = input.container === :m ? messages : marginals
        if input.key isa Tuple
            input.container === :q || throw(ArgumentError("`m[$(input.key)]`: a cluster has a marginal, not a message"))
            push!(collection[1], input.key)
            push!(collection[2], cluster_marginal(factornode, input.key))
        else
            for selection in selected
                push!(collection[1], input_label(factornode, selection))
                push!(collection[2], input.container === :m ? selection : getvariable(selection))
            end
            # A group selection of no members still reaches the rule, as a tuple of `nothing`s.
            isempty(selected) && !(input.selector isa MessagePassingRulesBase.SingleInterface) &&
                push!(collection[1], EmptyGroup(input.key, group_length(factornode, input.key)))
        end
    end
    return map(Tuple, messages), map(Tuple, marginals)
end

"""
    ReactiveMP.extended_default_dependencies(factornode, interface, inputs)

The dependencies of a target declared with `default`, `:a => (default, q[:a])`: the default
scheme's inputs, and each of `inputs`, a single interface's message or marginal, placed in
interface order. A marginal added this way is the variable's own, and an input the default
scheme has already is not added twice.
"""
function extended_default_dependencies(factornode, interface, inputs)
    iindex = findfirst(i -> i === interface, getinterfaces(factornode))
    (messagelabels, messages), (marginallabels, marginals) = default_dependencies(factornode, iindex)
    messages, messagelabels = collect(Any, messages), collect(Any, messagelabels)
    marginals, marginallabels = collect(Any, marginals), collect(Any, marginallabels)
    position(i) = findfirst(j -> j === i, getinterfaces(factornode))
    clusters = getlocalclusters(factornode)
    # The default scheme's marginals are the other clusters, in cluster order; each is placed by
    # the position of its first member.
    cindex = clusterindex(clusters, iindex)
    firsts = Any[first(getfactorization(clusters, c)) for c in eachindex(get_node_local_marginals(clusters)) if c != cindex]
    for input in inputs
        added = getinterface(factornode, interfaceindex(factornode, input.key))
        label = input_label(factornode, added)
        at = position(added)
        if input.container === :m
            label in messagelabels && continue
            k = count(i -> position(i) < at, messages) + 1
            insert!(messages, k, added)
            insert!(messagelabels, k, label)
        else
            label in marginallabels && continue
            k = count(<(at), firsts) + 1
            insert!(marginals, k, getvariable(added))
            insert!(marginallabels, k, label)
            insert!(firsts, k, at)
        end
    end
    return (Tuple(messagelabels), Tuple(messages)), (Tuple(marginallabels), Tuple(marginals))
end

function selected_interfaces(factornode, input::MessagePassingRulesBase.Dependency, interface)
    input.key isa Tuple && return ()
    input.selector isa MessagePassingRulesBase.SingleInterface && return (getinterface(factornode, interfaceindex(factornode, input.key)),)
    members = group_members(factornode, input.key)
    k = interface isa IndexedNodeInterface ? index(interface) : 0
    return map(i -> members[i], MessagePassingRulesBase.selected_indices(input.selector, k, length(members)))
end

function cluster_marginal(factornode, key::Tuple)
    marginals = get_node_local_marginals(getlocalclusters(factornode))
    position = findfirst(marginal -> name(marginal) == key, marginals)
    position === nothing && throw(
        ArgumentError("`$(functionalform(factornode))` consumes `q[$(join(repr.(key), ", "))]`, which is not a cluster of its factorisation"),
    )
    return marginals[position]
end

function activate_messages!(factornode, options)
    fform = functionalform(factornode)
    algorithm = getalgorithm(fform, options)
    annotations = getannotations(options)
    callbacks = getcallbacks(options)
    stream_postprocessor = getpostprocessor(options)
    spec = MessagePassingRulesBase.dependencies_spec(fform, algorithm)

    return foreach(enumerate(getinterfaces(factornode))) do (iindex, interface)
        if israndom(interface) || isdata(interface)
            (messagelabels, message_dependencies), (marginallabels, marginal_dependencies) =
                spec === nothing ? default_dependencies(factornode, iindex) : declared_dependencies(factornode, spec, interface)
            messagesnames, messages = collect_latest_messages(messagelabels, message_dependencies)
            marginalsnames, marginals = collect_latest_marginals(marginallabels, marginal_dependencies)

            stream_of_outbound_messages = with_statics(factornode, combineLatest((messages, marginals), PushNew()))

            mapping = let messagemap = MessageMapping(
                    fform, rule_target(interface), messagesnames, marginalsnames,
                    algorithm, annotations, factornode, callbacks, getdiagnostics(options), getcontext(options), getrulefallback(options), getlogscales(options),
                )
                (dependencies) -> DeferredMessage(dependencies[1], dependencies[2], messagemap)
            end

            stream_of_outbound_messages = stream_of_outbound_messages |> map(AbstractMessage, mapping)
            stream_of_outbound_messages = postprocess_stream_of_outbound_messages(
                stream_postprocessor, stream_of_outbound_messages
            )
            set_stream_of_outbound_messages!(interface, stream_of_outbound_messages)
        end
    end
end

"""
    ReactiveMP.rule_target(interface)

The target a message rule computes for `interface`:
[`Target`](@extref MessagePassingRulesBase.Target)`{:out}()`, or
[`IndexedTarget`](@extref MessagePassingRulesBase.IndexedTarget)`{:m}(k)` for member `k` of the
group `m`.
"""
rule_target(interface::NodeInterface) = MessagePassingRulesBase.Target{name(interface)}()
rule_target(interface::IndexedNodeInterface) = MessagePassingRulesBase.IndexedTarget{name(interface)}(index(interface))
