
"""
    FactorNodeLocalMarginal

The marginal of one cluster of a factor node's factorisation. Its `key` is what a rule reads
it by: the interface name for a single-interface cluster, `:μ`, and the tuple of member names
for a joint, `(:out, :μ)`. A single-interface cluster shares the stream of the variable's own
marginal.
"""
mutable struct FactorNodeLocalMarginal{K}
    const key::K
    marginal::MarginalObservable

    FactorNodeLocalMarginal(key::K) where {K <: Union{Symbol, Tuple{Vararg{Symbol}}}} = new{K}(key)
end

name(localmarginal::FactorNodeLocalMarginal) = localmarginal.key

get_stream_of_marginals(localmarginal::FactorNodeLocalMarginal) =
    localmarginal.marginal

function set_stream_of_marginals!(
        localmarginal::FactorNodeLocalMarginal, stream::MarginalObservable
    )
    return localmarginal.marginal = stream
end

function set_stream_of_marginals!(
        localmarginal::FactorNodeLocalMarginal, stream
    )
    marginal = MarginalObservable()
    connect!(marginal, stream)
    return localmarginal.marginal = marginal
end

Base.show(io::IO, marginal::FactorNodeLocalMarginal) =
    print(io, "FactorNodeLocalMarginal(", repr(name(marginal)), ")")

## FactorNodeLocalClusters

"""
    FactorNodeLocalClusters

The clusters of a factor node: the factorisation, as tuples of interface positions, and one
[`ReactiveMP.FactorNodeLocalMarginal`](@ref) per cluster.
"""
struct FactorNodeLocalClusters{M, F}
    marginals::M
    factorization::F
end

get_node_local_marginals(clusters::FactorNodeLocalClusters) = clusters.marginals
set_node_local_marginal_stream!(
    clusters::FactorNodeLocalClusters, index, stream
) = set_stream_of_marginals!(clusters.marginals[index], stream)

getfactorization(clusters::FactorNodeLocalClusters) = clusters.factorization
getfactorization(clusters::FactorNodeLocalClusters, index::Int) =
    clusters.factorization[index]

function FactorNodeLocalClusters(interfaces::Union{AbstractVector, Tuple}, factorization::Tuple)
    marginals = map(cluster -> FactorNodeLocalMarginal(clusterkey(cluster, interfaces)), factorization)
    return FactorNodeLocalClusters{typeof(marginals), typeof(factorization)}(marginals, factorization)
end

clusterindex(clusters::FactorNodeLocalClusters, vindex::Int) =
    findfirst(cluster -> vindex in cluster, clusters.factorization)

# Every local marginal but the one at `index`. The marginals are of different types once a
# joint is among them, so this cannot be `TupleTools.deleteat`, which wants an `NTuple`.
other_clusters(marginals::Tuple, index::Int) = Tuple(marginals[i] for i in eachindex(marginals) if i != index)

"""
    ReactiveMP.clusterkey(cluster, interfaces)

The key a cluster of interface positions is read by: the name of its only interface, or the
tuple of its members' names.
"""
clusterkey(cluster::Tuple{Int}, interfaces) = name(interfaces[only(cluster)])
clusterkey(cluster::Tuple{Vararg{Int}}, interfaces) = map(i -> name(interfaces[i]), cluster)

function initialize_clusters!(clusters::FactorNodeLocalClusters, factornode, options)
    # Every stream exists before any is wired, since a joint's rule may read any other cluster
    for i in eachindex(get_node_local_marginals(clusters))
        initialize_cluster!(clusters, i, factornode)
    end
    for i in eachindex(get_node_local_marginals(clusters))
        activate_cluster!(clusters, i, factornode, options)
    end
    return
end

function initialize_cluster!(clusters::FactorNodeLocalClusters, index::Int, factornode)
    localfactorization = getfactorization(clusters, index)
    stream_of_cluster_marginals = if isone(length(localfactorization))
        get_stream_of_marginals(getvariable(getinterface(factornode, first(localfactorization))))
    else
        MarginalObservable()
    end
    return set_node_local_marginal_stream!(clusters, index, stream_of_cluster_marginals)
end

function activate_cluster!(clusters::FactorNodeLocalClusters, index::Int, factornode, options)
    localfactorization = getfactorization(clusters, index)
    isone(length(localfactorization)) && return nothing

    marginal = get_node_local_marginals(clusters)[index]
    message_dependencies = map(i -> getinterface(factornode, i), localfactorization)
    marginal_dependencies = other_clusters(get_node_local_marginals(clusters), index)

    messagesnames, messages = collect_latest_messages(message_dependencies)
    marginalsnames, marginals = collect_latest_marginals(marginal_dependencies)

    fform = functionalform(factornode)
    mapping = MarginalMapping(
        fform,
        MessagePassingRulesBase.ClusterTarget(name(marginal)),
        messagesnames,
        marginalsnames,
        getalgorithm(fform, options),
        factornode,
    )
    marginalout = combineLatestUpdates((messages, marginals), PushNew(), Marginal, mapping, reset_vstatus)
    marginalout = postprocess_stream_of_marginals(getpostprocessor(options), marginalout)

    return set_stream_of_marginals!(marginal, marginalout)
end
