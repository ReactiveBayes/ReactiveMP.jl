
function message_dependencies end
function marginal_dependencies end

function functional_dependencies(dependencies::Any, factornode::GenericFactorNode, clusters::FactorNodeLocalClusters, interface::NodeInterface, iindex::Int)
    cindex = clusterindex(clusters, iindex)
    cluster = getfactorization(clusters, cindex)

    messages  = message_dependencies(dependencies, factornode, clusters, cluster, cindex, interface, iindex)
    marginals = marginal_dependencies(dependencies, factornode, clusters, cluster, cindex, interface, iindex)

    return messages, marginals
end

function get_messages_observable(factornode::GenericFactorNode, messages)
    if !isempty(messages)
        return get_messages_observable(factornode, Tuple(messages))
    else
        return nothing, of(nothing)
    end
end

function get_messages_observable(factornode::GenericFactorNode, messages::Tuple)
    return get_marginals_observable(factornode, marginals, Val{map(name, messages)}())
end

function get_messages_observable(::GenericFactorNode, messages::Tuple, messages_names::Val)
    messages_observable = combineLatestUpdates(map(m -> messagein(m), messages), PushNew())
    return messages_names, messages_observable
end

function get_marginals_observable(factornode::GenericFactorNode, marginals)
    if !isempty(marginals)
        return get_marginals_observable(factornode, Tuple(marginals))
    else
        return nothing, of(nothing)
    end
end

function get_marginals_observable(factornode::GenericFactorNode, marginals::Tuple)
    return get_marginals_observable(factornode, marginals, Val{map(name, marginals)}())
end

function get_marginals_observable(::GenericFactorNode, marginals::Tuple, marginal_names::Val)
    marginals_streams    = map(marginal -> getstream(marginal), marginals)
    marginals_observable = combineLatestUpdates(marginals_streams, PushNew())
    return marginal_names, marginals_observable
end



"""
    DefaultFunctionalDependencies

This functional dependencies translate directly to a regular variational message passing scheme. 
In order to compute a message out of some interface, this strategy requires messages from interfaces within the same cluster and marginals over other clusters.
"""
struct DefaultFunctionalDependencies end

function collect_functional_dependencies end

collect_functional_dependencies(T::Any, ::Nothing) = DefaultFunctionalDependencies()
collect_functional_dependencies(T::Any, something) = something

### Default

function message_dependencies(::DefaultFunctionalDependencies, factornode::GenericFactorNode, clusters::FactorNodeLocalClusters, cluster, cindex, interface, iindex)
    # First we remove current edge index from the list of dependencies
    vdependencies = filter(ci -> ci !== iindex, cluster)
    # Second we map interface indices to the actual interfaces
    return Iterators.map(inds -> map(i -> getinterface(factornode, i), inds), vdependencies)
end

function marginal_dependencies(::DefaultFunctionalDependencies, factornode::GenericFactorNode, clusters::FactorNodeLocalClusters, cluster, cindex, interface, iindex)
    # We skip the current cluster index
    return skipindex(getmarginals(clusters), cindex)
end

### With inbound messages
## old code that needs to be fixed 

"""
    RequireMessageFunctionalDependencies(indices::Tuple, start_with::Tuple)

The same as `DefaultFunctionalDependencies`, but in order to compute a message out of some edge also requires the inbound message on the this edge.

# Arguments

- `indices`::Tuple, tuple of integers, which indicates what edges should require inbound messages
- `start_with::Tuple`, tuple of `nothing` or `<:Distribution`, which specifies the initial inbound messages for edges in `indices`

Note: `start_with` uses `setmessage!` mechanism, hence, it can be visible by other listeners on the same edge. Explicit call to `setmessage!` overwrites whatever has been passed in `start_with`.

`@model` macro accepts a simplified construction of this pipeline:

```julia
@model function some_model()
    # ...
    y ~ NormalMeanVariance(x, τ) where {
        pipeline = RequireMessage(x = vague(NormalMeanPrecision),     τ)
                                  # ^^^                               ^^^
                                  # request 'inbound' for 'x'         we may do the same for 'τ',
                                  # and initialise with `vague(...)`  but here we skip initialisation
    }
    # ...
end
```

Deprecation warning: `RequireInboundFunctionalDependencies` has been deprecated in favor of `RequireMessageFunctionalDependencies`.

See also: [`ReactiveMP.DefaultFunctionalDependencies`](@ref), [`ReactiveMP.RequireMarginalFunctionalDependencies`](@ref), [`ReactiveMP.RequireEverythingFunctionalDependencies`](@ref)
"""
struct RequireMessageFunctionalDependencies{I, S}
    indices    :: I
    start_with :: S
end

Base.@deprecate_binding RequireInboundFunctionalDependencies RequireMessageFunctionalDependencies

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

### With marginals

"""
    RequireMarginalFunctionalDependencies(indices::Tuple, start_with::Tuple)

Similar to `DefaultFunctionalDependencies`, but in order to compute a message out of some edge also requires the posterior marginal on that edge.

# Arguments

- `indices`::Tuple, tuple of integers, which indicates what edges should require their own marginals
- `start_with::Tuple`, tuple of `nothing` or `<:Distribution`, which specifies the initial marginal for edges in `indices`

Note: `start_with` uses the `setmarginal!` mechanism, hence it can be visible to other listeners on the same edge. Explicit calls to `setmarginal!` overwrites whatever has been passed in `start_with`.

`@model` macro accepts a simplified construction of this pipeline:

```julia
@model function some_model()
    # ...
    y ~ NormalMeanVariance(x, τ) where {
        pipeline = RequireMarginal(x = vague(NormalMeanPrecision),     τ)
                                   # ^^^                               ^^^
                                   # request 'marginal' for 'x'        we may do the same for 'τ',
                                   # and initialise with `vague(...)`  but here we skip initialisation
    }
    # ...
end
```

Note: The simplified construction in `@model` macro syntax is only available in `GraphPPL.jl` of version `>2.2.0`.

See also: [`ReactiveMP.DefaultFunctionalDependencies`](@ref), [`ReactiveMP.RequireMessageFunctionalDependencies`](@ref), [`ReactiveMP.RequireEverythingFunctionalDependencies`](@ref)
"""
struct RequireMarginalFunctionalDependencies{I, S}
    indices    :: I
    start_with :: S
end

function message_dependencies(::RequireMarginalFunctionalDependencies, nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
    return message_dependencies(DefaultFunctionalDependencies(), nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
end

function marginal_dependencies(dependencies::RequireMarginalFunctionalDependencies, nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
    # First we find dependency index in `indices`, we use it later to find `start_with` distribution
    depindex = findfirst((i) -> i === iindex, dependencies.indices)

    if depindex !== nothing
        # We create an auxiliary local marginal with non-standard index here and inject it to other standard dependencies
        extra_localmarginal = FactorNodeLocalMarginal(-1, iindex, name(nodeinterfaces[iindex]))
        vmarginal           = getmarginal(connected_properties(nodeinterfaces[iindex]), IncludeAll())
        start_with          = dependencies.start_with[depindex]
        # Initialise now, if marginal has not been initialised before and `start_with` element is not empty
        if isnothing(getrecent(vmarginal)) && !isnothing(start_with)
            setmarginal!(vmarginal, start_with)
        end
        setstream!(extra_localmarginal, vmarginal)
        default = marginal_dependencies(DefaultFunctionalDependencies(), nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
        # Find insertion position (probably might be implemented more efficiently)
        insertafter = sum(first(el) < iindex ? 1 : 0 for el in default; init = 0)
        return TupleTools.insertafter(default, insertafter, (extra_localmarginal,))
    else
        return marginal_dependencies(DefaultFunctionalDependencies(), nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
    end
end

### Everything

"""
   RequireEverythingFunctionalDependencies

This pipeline specifies that in order to compute a message of some edge update rules request everything that is available locally.
This includes all inbound messages (including on the same edge) and marginals over all local edge-clusters (this may or may not include marginals on single edges, depends on the local factorisation constraint).

See also: [`DefaultFunctionalDependencies`](@ref), [`RequireMessageFunctionalDependencies`](@ref), [`RequireMarginalFunctionalDependencies`](@ref)
"""
struct RequireEverythingFunctionalDependencies end

function ReactiveMP.message_dependencies(::RequireEverythingFunctionalDependencies, nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
    # Return all node interfaces including the edge we are trying to compute a message on
    return nodeinterfaces
end

function ReactiveMP.marginal_dependencies(::RequireEverythingFunctionalDependencies, nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
    # Returns only local marginals based on local q factorisation, it does not return all possible combinations of all joint posterior marginals
    return nodelocalmarginals
end