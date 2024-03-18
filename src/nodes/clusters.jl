
"""
    FactorNodeLocalMarginal

This object represents local marginals for some specific factor node.
The local marginal can be joint in case of structured factorisation.
Local to factor node marginal also can be shared with a corresponding marginal of some random variable.

See also: [`FactorNodeLocalClusters`](@ref)
"""
mutable struct FactorNodeLocalMarginal
    const name::Symbol
    marginal::MarginalObservable

    FactorNodeLocalMarginal(name::Symbol) = new(name)
end

name(localmarginal::FactorNodeLocalMarginal) = localmarginal.name
tag(localmarginal::FactorNodeLocalMarginal)  = Val{name(localmarginal)}()

getmarginal(localmarginal::FactorNodeLocalMarginal) = localmarginal.marginal
setmarginal!(localmarginal::FactorNodeLocalMarginal, marginal) = localmarginal.marginal = marginal

Base.show(io::IO, marginal::FactorNodeLocalMarginal) = print(io, "FactorNodeLocalMarginal(", name(marginal), ")")

## FactorNodeLocalClusters

struct FactorNodeLocalClusters{M, F}
    marginals::M
    factorization::F
end

getmarginals(clusters::FactorNodeLocalClusters) = clusters.marginals
getmarginal(clusters::FactorNodeLocalClusters, index) = getindex(getmarginals(clusters), index)
setmarginal!(clusters::FactorNodeLocalClusters, index, marginal::MarginalObservable) = setmarginal!(getmarginal(clusters, index), marginal)

getfactorization(clusters::FactorNodeLocalClusters) = clusters.factorization
getfactorization(clusters::FactorNodeLocalClusters, index::Int) = clusters.factorization[index]

function FactorNodeLocalClusters(interfaces::AbstractArray{NodeInterface}, factorization)
    marginals = map(factor -> FactorNodeLocalMarginal(clustername(factor, interfaces)), factorization)
    return FactorNodeLocalClusters(marginals, factorization)
end

function FactorNodeLocalClusters(interfaces::NTuple{N, NodeInterface}, factorization) where {N}
    marginals = map(factor -> FactorNodeLocalMarginal(clustername(factor, interfaces)), factorization)
    return FactorNodeLocalClusters(marginals, factorization)
end

## FactorNodeLocalCluster

clusterindex(clusters::FactorNodeLocalClusters, vindex::Int) = clusterindex(clusters, clusters.factorization, vindex)
clusterindex(::FactorNodeLocalClusters, factorization, vindex::Int) = findfirst(cluster -> vindex in cluster, factorization)

clustername(cluster::Tuple, interfaces) = mapreduce(v -> name(interfaces[v]), (a, b) -> Symbol(a, :_, b), cluster)
clustername(cluster, interfaces) = reduce((a, b) -> Symbol(a, :_, b), Iterators.map(v -> name(interfaces[v]), cluster))
clustername(interfaces) = reduce((a, b) -> Symbol(a, :_, b), Iterators.map(interface -> name(interface), interfaces))

function initialize_clusters!(clusters::FactorNodeLocalClusters, dependencies, factornode, options)
    # We first need to initialize all the clusters, since the `activate_cluster!` function may use any of the marginals
    for i in eachindex(getmarginals(clusters))
        initialize_cluster!(clusters, i, dependencies, factornode, options)
    end
    for i in eachindex(getmarginals(clusters))
        activate_cluster!(clusters, i, dependencies, factornode, options)
    end
end

function initialize_cluster!(clusters::FactorNodeLocalClusters, index::Int, dependencies, factornode, options)
    localfactorization = getfactorization(clusters, index)
    # For the clusters of length `1` there is no need to create a new `MarginalObservable` object
    # We can simply reuse it from the variable connected to the factor node. Potentially it saves a bit of memory 
    stream_of_cluster_marginals = if isone(length(localfactorization))
        getmarginal(getvariable(getinterface(factornode, first(localfactorization))), IncludeAll())
    else
        # For the clusters of length `>1` we need to create the new strean, but it will be assigned later
        MarginalObservable()
    end
    setmarginal!(clusters, index, stream_of_cluster_marginals)
end

function activate_cluster!(clusters::FactorNodeLocalClusters, index::Int, dependencies, factornode, options)
    localfactorization = getfactorization(clusters, index)

    if !isone(length(localfactorization))
        # For the clusters which length is not equal to one we should collect the dependencies 
        # and call the `MarginalMapping` to compute the result. The `MarginalObservable` should have 
        # been initialized in the `initialize_cluster!` before
        marginal = getmarginal(clusters, index)

        clusterinterfaces = map(i -> getinterface(factornode, i), localfactorization)

        message_dependencies  = tuple(clusterinterfaces...)
        marginal_dependencies = tuple(TupleTools.deleteat(getmarginals(clusters), index)...)

        messagestag, messages = collect_latest_messages(dependencies, factornode, message_dependencies)
        marginalstag, marginals = collect_latest_marginals(dependencies, factornode, marginal_dependencies)

        fform = functionalform(factornode)
        vtag  = tag(getmarginal(clusters, index))
        meta  = collect_meta(fform, getmetadata(options))

        mapping = MarginalMapping(fform, vtag, messagestag, marginalstag, meta, node_if_required(fform, factornode))
        # TODO: discontinue operator is needed for loopy belief propagation? Check
        marginalout = combineLatest((messages, marginals), PushNew()) |> discontinue() |> map(Marginal, mapping)

        connect!(getmarginal(marginal), marginalout)
    end
end