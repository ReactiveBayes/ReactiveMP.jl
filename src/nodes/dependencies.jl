# What each outbound message is computed from, and the streams that deliver it.

collect_latest_messages(collection) =
    collect_latest_updates(get_stream_of_inbound_messages, nothing, collection)
collect_latest_marginals(collection) =
    collect_latest_updates(get_stream_of_marginals, reset_vstatus_of_sources, collection)

collect_latest_updates(f::F, callback::C, collection) where {F, C} =
    collect_latest_updates(f, callback, Tuple(collection))

function collect_latest_updates(f::F, callback::C, collection::Tuple) where {F, C}
    return if isempty(collection)
        (nothing, of(nothing))
    else
        streams = map(f, collection)
        (
            Val{map(name, collection)}(),
            combineLatestUpdates(streams, PushNew(), typeof(streams), identity, callback),
        )
    end
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
    ReactiveMP.default_dependencies(factornode, iindex)

The engine's default scheme, a regular variational message passing scheme driven by the
factorisation: the message out of interface `iindex` is computed from the inbound messages of
the other interfaces in its cluster, and the marginals of every other cluster.
"""
function default_dependencies(factornode, iindex)
    clusters = getlocalclusters(factornode)
    cindex = clusterindex(clusters, iindex)
    cluster = getfactorization(clusters, cindex)
    message_dependencies = map(i -> getinterface(factornode, i), filter(i -> i !== iindex, cluster))
    marginal_dependencies = other_clusters(get_node_local_marginals(clusters), cindex)
    return message_dependencies, marginal_dependencies
end

function activate_messages!(factornode, options)
    fform = functionalform(factornode)
    algorithm = getalgorithm(fform, options)
    annotations = getannotations(options)
    callbacks = getcallbacks(options)
    stream_postprocessor = getpostprocessor(options)

    return foreach(enumerate(getinterfaces(factornode))) do (iindex, interface)
        if israndom(interface) || isdata(interface)
            message_dependencies, marginal_dependencies = default_dependencies(factornode, iindex)
            messagesnames, messages = collect_latest_messages(message_dependencies)
            marginalsnames, marginals = collect_latest_marginals(marginal_dependencies)

            stream_of_outbound_messages = combineLatest((messages, marginals), PushNew())

            mapping = let messagemap = MessageMapping(
                    fform, rule_target(interface), messagesnames, marginalsnames,
                    algorithm, annotations, factornode, callbacks,
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

The target a message rule computes for `interface`: `Target{:out}()`, or
`IndexedTarget{:m}(k)` for member `k` of a group.
"""
rule_target(interface::NodeInterface) = MessagePassingRulesBase.Target{name(interface)}()
rule_target(interface::IndexedNodeInterface) = MessagePassingRulesBase.IndexedTarget{name(interface)}(index(interface))
