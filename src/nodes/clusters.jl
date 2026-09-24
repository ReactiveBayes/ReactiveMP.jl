
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

    FactorNodeLocalMarginal(key::K) where {K <: Union{Symbol, Tuple{Vararg{Union{Symbol, Tuple{Symbol, Int}}}}}} = new{K}(key)
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
tuple of its members' names, in which a group whose members are all in the cluster appears
once, by its name: `(:in,)` is the joint over the group `in`, even of one member. A joint that
holds only some of a group's members names each, `(group, index)`: `(:out, (:T, 1))`. A cluster
with a key that is a name shares its variable's marginal; one with a tuple is a joint.
"""
function clusterkey(cluster::Tuple{Vararg{Int}}, interfaces)
    members = map(i -> interfaces[i], cluster)
    names = Union{Symbol, Tuple{Symbol, Int}}[]
    for member in members
        if member isa IndexedNodeInterface && whole_group(member, members, interfaces)
            name(member) in names || push!(names, name(member))
        elseif member isa IndexedNodeInterface && length(cluster) > 1
            push!(names, interface_key(member))
        else
            push!(names, name(member))
        end
    end
    return isone(length(cluster)) && isone(length(names)) && !(first(members) isa IndexedNodeInterface && whole_group(first(members), members, interfaces)) ?
        only(names) : Tuple(names)
end

# Whether every member of `member`'s group is among `members`.
whole_group(member::IndexedNodeInterface, members, interfaces) =
    count(i -> i isa IndexedNodeInterface && name(i) === name(member), members) ==
    count(i -> i isa IndexedNodeInterface && name(i) === name(member), interfaces)

isjoint(localmarginal::FactorNodeLocalMarginal) = name(localmarginal) isa Tuple

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
    stream_of_cluster_marginals = if !isjoint(get_node_local_marginals(clusters)[index])
        get_stream_of_marginals(getvariable(getinterface(factornode, only(localfactorization))))
    else
        MarginalObservable()
    end
    return set_node_local_marginal_stream!(clusters, index, stream_of_cluster_marginals)
end

# A joint is computed by the node's marginal rule. In a stochastic node it reads the messages of
# its members and the marginals of the other clusters. In a deterministic node, whose output is
# a function of its inputs, the joint over the inputs reads the messages on every interface, the
# output's included (v6's `q_ins`).
function activate_cluster!(clusters::FactorNodeLocalClusters, index::Int, factornode, options)
    marginal = get_node_local_marginals(clusters)[index]
    isjoint(marginal) || return nothing

    others = isdeterministic(sdtype(factornode)) ? () : Tuple(i for i in eachindex(get_node_local_marginals(clusters)) if i != index)
    message_dependencies = isdeterministic(sdtype(factornode)) ? Tuple(getinterfaces(factornode)) :
        map(i -> getinterface(factornode, i), getfactorization(clusters, index))

    messagesnames, messages = collect_latest_messages(map(i -> input_label(factornode, i), message_dependencies), message_dependencies)
    marginalsnames, marginals = collect_latest_marginals(
        map(i -> cluster_label(factornode, clusters, i), others), map(i -> get_node_local_marginals(clusters)[i], others),
    )
    messages, marginals = with_statics(factornode, messages), with_statics(factornode, marginals)

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
