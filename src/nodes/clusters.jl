
"""
    FactorNodeLocalMarginal

This object represents local marginals for some specific factor node.
The local marginal can be joint in case of structured factorisation.
Local to factor node marginal also can be shared with a corresponding marginal of some random variable.

See also: [`FactorNodeLocalClusters`](@ref)
"""
mutable struct FactorNodeLocalMarginal
    const name::Symbol
    stream::MarginalObservable

    FactorNodeLocalMarginal(name::Symbol) = new(name)
end

name(localmarginal::FactorNodeLocalMarginal) = localmarginal.name
tag(localmarginal::FactorNodeLocalMarginal)  = Val{name(localmarginal)}()

getstream(localmarginal::FactorNodeLocalMarginal)              = localmarginal.stream
setstream!(localmarginal::FactorNodeLocalMarginal, observable) = localmarginal.stream = observable

Base.show(io::IO, marginal::FactorNodeLocalMarginal) = print(io, "FactorNodeLocalMarginal(", name(marginal), ")")

## FactorNodeLocalClusters

struct FactorNodeLocalClusters{M, F}
    marginals::M
    factorization::F
end

getmarginals(clusters::FactorNodeLocalClusters) = clusters.marginals
getmarginal(clusters::FactorNodeLocalClusters, index) = getindex(getmarginals(clusters), index)

getfactorization(clusters::FactorNodeLocalClusters) = clusters.factorization
getfactorization(clusters::FactorNodeLocalClusters, index::Int) = clusters.factorization[index]

function FactorNodeLocalClusters(interfaces::NTuple{N, NodeInterface}, factorization::NTuple) where {N}
    marginals = ntuple(i -> FactorNodeLocalMarginal(clustername(factorization[i], interfaces)), length(factorization))
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