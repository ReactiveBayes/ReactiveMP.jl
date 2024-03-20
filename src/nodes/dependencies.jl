export DefaultFunctionalDependencies, RequireMessageFunctionalDependencies, RequireMarginalFunctionalDependencies, RequireEverythingFunctionalDependencies

collect_latest_messages(dependencies, factornode, collection) = __collect_latest_updates(messagein, collection)
collect_latest_marginals(dependencies, factornode, collection) = __collect_latest_updates(getmarginal, collection)

function __collect_latest_updates(f::F, collection) where {F}
    return __collect_latest_updates(f, Tuple(collection))
end

function __collect_latest_updates(f::F, collection::Tuple) where {F}
    return isempty(collection) ? (nothing, of(nothing)) : (Val{map(name, collection)}(), combineLatestUpdates(map(f, collection), PushNew()))
end

abstract type FunctionalDependencies end

function activate!(dependencies::FunctionalDependencies, factornode, options)
    scheduler = getscheduler(options)
    addons    = getaddons(options)
    fform     = functionalform(factornode)
    meta      = collect_meta(fform, getmetadata(options))
    pipeline  = collect_pipeline(fform, getpipeline(options))

    foreach(enumerate(getinterfaces(factornode))) do (iindex, interface)
        if israndom(interface) || isdata(interface)
            with_functional_dependencies(dependencies, factornode, interface, iindex) do message_dependencies, marginal_dependencies
                messagestag, messages = collect_latest_messages(dependencies, factornode, message_dependencies)
                marginalstag, marginals = collect_latest_marginals(dependencies, factornode, marginal_dependencies)

                vtag        = tag(interface)
                vconstraint = Marginalisation()

                vmessageout = combineLatest((messages, marginals), PushNew())

                mapping = let messagemap = MessageMapping(fform, vtag, vconstraint, messagestag, marginalstag, meta, addons, node_if_required(fform, factornode))
                    (dependencies) -> VariationalMessage(dependencies[1], dependencies[2], messagemap)
                end

                vmessageout = vmessageout |> map(AbstractMessage, mapping)
                vmessageout = apply_pipeline_stage(pipeline, factornode, vtag, vmessageout)
                vmessageout = vmessageout |> schedule_on(scheduler)

                connect!(messageout(interface), vmessageout)
            end
        end
    end
end

function functional_dependencies end

function with_functional_dependencies(callback::F, strategy::FunctionalDependencies, factornode, interface, iindex) where {F}
    message_dependencies, marginal_dependencies = functional_dependencies(strategy, factornode, interface, iindex)
    return callback(message_dependencies, marginal_dependencies)
end

"""
    DefaultFunctionalDependencies

This functional dependencies translate directly to a regular variational message passing scheme. 
In order to compute a message out of some interface, this strategy requires messages from interfaces within the same cluster and marginals over other clusters.
"""
struct DefaultFunctionalDependencies <: FunctionalDependencies end

function collect_functional_dependencies end

collect_functional_dependencies(fform::F, ::Nothing) where {F} = default_functional_dependencies(fform)
collect_functional_dependencies(fform::F, something) where {F} = something

default_functional_dependencies(any) = DefaultFunctionalDependencies()

function functional_dependencies(::DefaultFunctionalDependencies, factornode, interface, iindex)
    clusters = getlocalclusters(factornode)
    # Find the index of the cluster for the current interface
    cindex = clusterindex(clusters, iindex)
    # Fetch the actual cluster
    cluster = getfactorization(clusters, cindex)
    # Remove current edge index from the list of dependencies in the given cluster
    vdependencies = filter(ci -> ci !== iindex, cluster)
    # Map interface indices to the actual interfaces to get the messages dependencies
    message_dependencies = Iterators.map(inds -> map(i -> getinterface(factornode, i), inds), vdependencies)

    # For the marginal dependencies we need to skip the current cluster
    marginal_dependencies = skipindex(getmarginals(clusters), cindex)

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
struct RequireMessageFunctionalDependencies{S <: NamedTuple} <: FunctionalDependencies
    specification::S
end

RequireMessageFunctionalDependencies(; kwargs...) = RequireMessageFunctionalDependencies((; kwargs...))

function message_dependencies(dependencies::RequireMessageFunctionalDependencies, nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)

    # First we find dependency index in `indices`, we use it later to find `start_with` distribution
    depindex = findfirst((i) -> i === iindex, dependencies.indices)

    # If we have `depindex` in our `indices` we include it in our list of functional dependencies. It effectively forces rule to require inbound message
    if depindex !== nothing
        # `mapindex` is a lambda function here
        output     = messagein(nodeinterfaces[iindex])
        start_with = dependencies.start_with[depindex]
        # Initialise now, if message has not been initialised before and `start_with` element is not empty
        if isnothing(getrecent(output)) && !isnothing(start_with)
            setmessage!(output, start_with)
        end
        return map(inds -> map(i -> @inbounds(nodeinterfaces[i]), inds), varcluster)
    else
        return message_dependencies(DefaultFunctionalDependencies(), nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
    end
end

function marginal_dependencies(::RequireMessageFunctionalDependencies, nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
    return marginal_dependencies(DefaultFunctionalDependencies(), nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
end

function functional_dependencies(dependencies::RequireMessageFunctionalDependencies, factornode, interface, iindex)
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
            setmessage!(messagein(interface), initialmessage)
        end
        # And return the cluster as is
        cluster
    end

    # Map interface indices to the actual interfaces to get the messages dependencies
    message_dependencies = Iterators.map(inds -> map(i -> getinterface(factornode, i), inds), vdependencies)

    # For the marginal dependencies we need to skip the current cluster
    marginal_dependencies = skipindex(getmarginals(clusters), cindex)

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
struct RequireMarginalFunctionalDependencies{S <: NamedTuple}
    specification::S
end

RequireMarginalFunctionalDependencies(; kwargs...) = RequireMarginalFunctionalDependencies((; kwargs...))

function functional_dependencies(dependencies::RequireMarginalFunctionalDependencies, factornode, interface, iindex)
    specification = dependencies.specification

    clusters = getlocalclusters(factornode)
    # Find the index of the cluster for the current interface
    cindex = clusterindex(clusters, iindex)
    # Fetch the actual cluster
    cluster = getfactorization(clusters, cindex)
    # Remove current edge index from the list of dependencies in the given cluster
    vdependencies = filter(ci -> ci !== iindex, cluster)
    # Map interface indices to the actual interfaces to get the messages dependencies
    message_dependencies = Iterators.map(inds -> map(i -> getinterface(factornode, i), inds), vdependencies)

    # For the marginal dependencies we need to skip the current cluster
    marginal_dependencies_default = skipindex(getmarginals(clusters), cindex)

    marginal_dependencies = if name(interface) ∈ keys(specification)
        # We create an auxiliary local marginal with non-standard index here and inject it to other standard dependencies
        extra_localmarginal = FactorNodeLocalMarginal(name(interface))
        # Create a stream of marginals and connect it with the streams of marginals of the actual variable
        extra_stream = MarginalObservable()
        connect!(extra_stream, getmarginal(getvariable(interface), IncludeAll()))
        setmarginal!(extra_localmarginal, extra_stream)

        initialmarginals = specification[name(interface)]
        if !isnothing(initialmarginals)
            setmarginal!(extra_stream, initialmarginals)
        end

        insertafter = sum(first(el) < iindex ? 1 : 0 for el in marginal_dependencies_default; init = 0)
        TupleTools.insertafter(marginal_dependencies_default, insertafter, (extra_localmarginal,))
    else
        marginal_dependencies_default
    end

    return message_dependencies, marginal_dependencies
end

"""
   RequireEverythingFunctionalDependencies

This pipeline specifies that in order to compute a message of some edge update rules request everything that is available locally.
This includes all inbound messages (including on the same edge) and marginals over all local edge-clusters (this may or may not include marginals on single edges, depends on the local factorisation constraint).

See also: [`DefaultFunctionalDependencies`](@ref), [`RequireMessageFunctionalDependencies`](@ref), [`RequireMarginalFunctionalDependencies`](@ref)
"""
struct RequireEverythingFunctionalDependencies end

function functional_dependencies(::RequireEverythingFunctionalDependencies, factornode, interface, iindex)
    clusters = getlocalclusters(factornode)
    # Find the index of the cluster for the current interface
    cindex = clusterindex(clusters, iindex)
    # Fetch the actual cluster
    cluster = getfactorization(clusters, cindex)

    message_dependencies = Iterators.map(inds -> map(i -> getinterface(factornode, i), inds), cluster)
    marginal_dependencies = getmarginals(clusters)

    return message_dependencies, marginal_dependencies
end