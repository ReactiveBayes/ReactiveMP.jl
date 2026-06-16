export DefaultFunctionalDependencies,
    RequireMessageFunctionalDependencies,
    RequireMarginalFunctionalDependencies,
    RequireEverythingFunctionalDependencies

collect_latest_messages(dependencies, factornode, collection) =
    __collect_latest_updates(get_stream_of_inbound_messages, collection)
collect_latest_marginals(dependencies, factornode, collection) =
    __collect_latest_updates(get_stream_of_marginals, collection)

function __collect_latest_updates(f::F, collection) where {F}
    return __collect_latest_updates(f, Tuple(collection))
end

function __collect_latest_updates(f::F, collection::Tuple) where {F}
    return if isempty(collection)
        (nothing, of(nothing))
    else
        (
            Val{map(name, collection)}(),
            combineLatestUpdates(map(f, collection), PushNew()),
        )
    end
end

"""
    ReactiveMP.FunctionalDependencies

Abstract supertype for policies that determine which messages and marginals are required to compute each outbound message at a factor node. A concrete subtype is passed as `options.dependencies` in [`ReactiveMP.FactorNodeActivationOptions`](@ref) and consulted during [`ReactiveMP.activate!(::FactorNode, ::FactorNodeActivationOptions)`](@ref).

See also: [`ReactiveMP.DefaultFunctionalDependencies`](@ref), [`ReactiveMP.RequireMessageFunctionalDependencies`](@ref), [`ReactiveMP.RequireMarginalFunctionalDependencies`](@ref), [`ReactiveMP.RequireEverythingFunctionalDependencies`](@ref)
"""
abstract type FunctionalDependencies end

function activate!(dependencies::FunctionalDependencies, factornode, options)
    annotations          = getannotations(options)
    rulefallback         = getrulefallback(options)
    callbacks            = getcallbacks(options)
    fform                = functionalform(factornode)
    meta                 = collect_meta(fform, getmetadata(options))
    stream_postprocessor = getpostprocessor(options)

    foreach(enumerate(getinterfaces(factornode))) do (iindex, interface)
        if israndom(interface) || isdata(interface)
            with_functional_dependencies(
                dependencies, factornode, interface, iindex
            ) do message_dependencies, marginal_dependencies
                messagestag, messages = collect_latest_messages(
                    dependencies, factornode, message_dependencies
                )
                marginalstag, marginals = collect_latest_marginals(
                    dependencies, factornode, marginal_dependencies
                )

                vtag        = tag(interface)
                vconstraint = Marginalisation()

                stream_of_outbound_messages = combineLatest(
                    (messages, marginals), PushNew()
                )

                mapping =
                    let messagemap = MessageMapping(
                            fform,
                            vtag,
                            vconstraint,
                            messagestag,
                            marginalstag,
                            meta,
                            annotations,
                            node_if_required(fform, factornode),
                            rulefallback,
                            callbacks,
                        )
                        (dependencies) -> DeferredMessage(
                            dependencies[1], dependencies[2], messagemap
                        )
                    end

                stream_of_outbound_messages =
                    stream_of_outbound_messages |> map(AbstractMessage, mapping)
                stream_of_outbound_messages = postprocess_stream_of_outbound_messages(
                    stream_postprocessor, stream_of_outbound_messages
                )
                set_stream_of_outbound_messages!(
                    interface, stream_of_outbound_messages
                )
            end
        end
    end
end

function functional_dependencies end

function with_functional_dependencies(
    callback::F, strategy::FunctionalDependencies, factornode, interface, iindex
) where {F}
    message_dependencies, marginal_dependencies = functional_dependencies(
        strategy, factornode, interface, iindex
    )
    return callback(message_dependencies, marginal_dependencies)
end

"""
    DefaultFunctionalDependencies

This functional dependencies translate directly to a regular variational message passing scheme. 
In order to compute a message out of some interface, this strategy requires messages from interfaces within the same cluster and marginals over other clusters.
"""
struct DefaultFunctionalDependencies <: FunctionalDependencies end

"""
    ReactiveMP.collect_functional_dependencies(fform, dependencies)

Returns the [`ReactiveMP.FunctionalDependencies`](@ref) instance to use for a factor node with functional form `fform`.
If `dependencies` is `nothing`, falls back to `default_functional_dependencies(fform)`, which returns [`ReactiveMP.DefaultFunctionalDependencies`](@ref) for most nodes.
Otherwise returns `dependencies` unchanged, allowing callers to override the policy per node.
"""
function collect_functional_dependencies end

collect_functional_dependencies(fform::F, ::Nothing) where {F} =
    default_functional_dependencies(fform)
collect_functional_dependencies(fform::F, something) where {F} = something

default_functional_dependencies(any) = DefaultFunctionalDependencies()

function functional_dependencies(
    ::DefaultFunctionalDependencies, factornode, interface, iindex
)
    clusters = getlocalclusters(factornode)
    # Find the index of the cluster for the current interface
    cindex = clusterindex(clusters, iindex)
    # Fetch the actual cluster
    cluster = getfactorization(clusters, cindex)
    # Remove current edge index from the list of dependencies in the given cluster
    vdependencies = filter(ci -> ci !== iindex, cluster)
    # Map interface indices to the actual interfaces to get the messages dependencies
    message_dependencies = Iterators.map(
        inds -> map(i -> getinterface(factornode, i), inds), vdependencies
    )

    # For the marginal dependencies we need to skip the current cluster
    marginal_dependencies = skipindex(
        get_node_local_marginals(clusters), cindex
    )

    return message_dependencies, marginal_dependencies
end

"""
    RequireMessageFunctionalDependencies(specifications::NamedTuple)
    RequireMessageFunctionalDependencies(; specifications...)

The same as `DefaultFunctionalDependencies`, but in order to compute a message out of some edge also requires the inbound message on the this edge.

The specification parameter is a named tuple that contains the names of the edges and their initial messages. 
When a name is present in the named tuple, that indicates that the computation of the outbound message on the same edge must use the inbound message.
If `nothing` is passed as a value in the named tuple, the initial message is not set. Note that the construction allows passing keyword argument to the constructor 
instead of using `NamedTuple` directly.

```julia
RequireMessageFunctionalDependencies(μ = vague(NormalMeanPrecision),     τ = nothing)
                                     # ^^^                               ^^^
                                     # request 'inbound' for 'x'         we may do the same for 'τ',
                                     # and initialise with `vague(...)`  but here we skip initialisation
```

See also: [`ReactiveMP.DefaultFunctionalDependencies`](@ref), [`ReactiveMP.RequireMarginalFunctionalDependencies`](@ref), [`ReactiveMP.RequireEverythingFunctionalDependencies`](@ref)
"""
struct RequireMessageFunctionalDependencies{S <: NamedTuple} <:
       FunctionalDependencies
    specification::S
end

RequireMessageFunctionalDependencies(; kwargs...) =
    RequireMessageFunctionalDependencies((; kwargs...))

function functional_dependencies(
    dependencies::RequireMessageFunctionalDependencies,
    factornode,
    interface,
    iindex,
)
    specification = dependencies.specification

    clusters = getlocalclusters(factornode)
    # Find the index of the cluster for the current interface
    cindex = clusterindex(clusters, iindex)
    # Fetch the actual cluster
    cluster = getfactorization(clusters, cindex)
    # Remove current edge index from the list of dependencies in the given cluster
    # only if its not in the specification for `RequireMessageFunctionalDependencies`
    # otherwise keep it and initialize with a message if the value of the specification is not nothing
    vdependencies = if name(interface) ∉ keys(specification)
        filter(ci -> ci !== iindex, cluster)
    else
        initialmessage = specification[name(interface)]
        # Set the initial message if its not `nothing`
        if !isnothing(initialmessage)
            set_initial_message!(
                get_stream_of_inbound_messages(interface), initialmessage
            )
        end
        # And return the cluster as is
        cluster
    end

    # Map interface indices to the actual interfaces to get the messages dependencies
    message_dependencies = Iterators.map(
        inds -> map(i -> getinterface(factornode, i), inds), vdependencies
    )

    # For the marginal dependencies we need to skip the current cluster
    marginal_dependencies = skipindex(
        get_node_local_marginals(clusters), cindex
    )

    return message_dependencies, marginal_dependencies
end

"""
    RequireMarginalFunctionalDependencies(specifications::NamedTuple)
    RequireMarginalFunctionalDependencies(; specifications...)

The same as `DefaultFunctionalDependencies`, but in order to compute a message out of some edge also requires the marginal on the this edge.

The specification parameter is a named tuple that contains the names of the edges and their initial marginals. 
When a name is present in the named tuple, that indicates that the computation of the outbound message on the same edge must use the marginal on this edge.
If `nothing` is passed as a value in the named tuple, the initial marginal is not set. Note that the construction allows passing keyword argument to the constructor 
instead of using `NamedTuple` directly.

```julia
RequireMarginalFunctionalDependencies(μ = vague(NormalMeanPrecision),     τ = nothing)
                                     # ^^^                               ^^^
                                     # request 'marginal' for 'x'        we may do the same for 'τ',
                                     # and initialise with `vague(...)`  but here we skip initialisation
```

See also: [`ReactiveMP.DefaultFunctionalDependencies`](@ref), [`ReactiveMP.RequireMessageFunctionalDependencies`](@ref), [`ReactiveMP.RequireEverythingFunctionalDependencies`](@ref)
"""
struct RequireMarginalFunctionalDependencies{S <: NamedTuple} <:
       FunctionalDependencies
    specification::S
end

RequireMarginalFunctionalDependencies(; kwargs...) =
    RequireMarginalFunctionalDependencies((; kwargs...))

function functional_dependencies(
    dependencies::RequireMarginalFunctionalDependencies,
    factornode,
    interface,
    iindex,
)
    specification = dependencies.specification

    clusters = getlocalclusters(factornode)
    # Find the index of the cluster for the current interface
    cindex = clusterindex(clusters, iindex)
    # Fetch the actual cluster
    cluster = getfactorization(clusters, cindex)
    # Remove current edge index from the list of dependencies in the given cluster
    vdependencies = filter(ci -> ci !== iindex, cluster)
    # Map interface indices to the actual interfaces to get the messages dependencies
    message_dependencies = Iterators.map(
        inds -> map(i -> getinterface(factornode, i), inds), vdependencies
    )

    # For the marginal dependencies we need to skip the current cluster
    marginal_dependencies_default_clusters      = skipindex(get_node_local_marginals(clusters), cindex)
    marginal_dependencies_default_factorization = skipindex(getfactorization(clusters), cindex)

    marginal_dependencies = if name(interface) ∈ keys(specification)

        # We create an auxiliary local marginal with non-standard index here and inject it to other standard dependencies
        extra_localmarginal = FactorNodeLocalMarginal(name(interface))
        # Create a stream of marginals and connect it with the streams of marginals of the actual variable
        extra_stream = MarginalObservable()
        connect!(extra_stream, get_stream_of_marginals(getvariable(interface)))
        set_stream_of_marginals!(extra_localmarginal, extra_stream)

        initialmarginals = specification[name(interface)]
        if !isnothing(initialmarginals)
            set_initial_marginal!(extra_stream, initialmarginals)
        end

        insertafter = sum(
            first(el) < iindex ? 1 : 0 for
            el in marginal_dependencies_default_factorization;
            init = 0,
        )
        TupleTools.insertafter(
            marginal_dependencies_default_clusters,
            insertafter,
            (extra_localmarginal,),
        )
    else
        marginal_dependencies_default_clusters
    end

    return message_dependencies, marginal_dependencies
end

"""
   RequireEverythingFunctionalDependencies

This strategy specifies that in order to compute a message of some edge update rules request everything that is available locally.
This includes all inbound messages (including on the same edge) and marginals over all local edge-clusters (this may or may not include marginals on single edges, depends on the local factorisation constraint).

See also: [`DefaultFunctionalDependencies`](@ref), [`RequireMessageFunctionalDependencies`](@ref), [`RequireMarginalFunctionalDependencies`](@ref)
"""
struct RequireEverythingFunctionalDependencies <: FunctionalDependencies end

function functional_dependencies(
    ::RequireEverythingFunctionalDependencies, factornode, interface, iindex
)
    clusters = getlocalclusters(factornode)
    # Find the index of the cluster for the current interface
    cindex = clusterindex(clusters, iindex)
    # Fetch the actual cluster
    cluster = getfactorization(clusters, cindex)

    message_dependencies = Iterators.map(
        inds -> map(i -> getinterface(factornode, i), inds), cluster
    )
    marginal_dependencies = get_node_local_marginals(clusters)

    return message_dependencies, marginal_dependencies
end
