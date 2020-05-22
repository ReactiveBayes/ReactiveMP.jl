export NodeVariable, nodevar, name, messageout, messagein
export Node, functionalform, variables, factorisation, factors, varindex, iscontain, isfactorised
export getcluster, clusters, clusterindex
export deps

using StaticArrays
using BenchmarkTools
using Rocket

import Base: show

struct NodeVariable
    name  :: Symbol
    m_out :: LazyObservable{Message}
    m_in  :: LazyObservable{Message}

    NodeVariable(name::Symbol) = new(name, lazy(Message), lazy(Message))
end

Base.show(io::IO, nodevar::NodeVariable) = print(io, name(nodevar))

nodevar(name::Symbol) = NodeVariable(name)

name(nodevar::NodeVariable)       = nodevar.name
messageout(nodevar::NodeVariable) = nodevar.m_out
messagein(nodevar::NodeVariable)  = nodevar.m_in

struct Node{F, N, C}
    variables     :: SVector{N, NodeVariable}
    factorisation :: C
end

function Node(::Type{F}, variables::SVector{N, NodeVariable}, factorisation::C) where { F, N, C }
    return Node{F, N, C}(deepcopy(variables), deepcopy(factorisation))
end

function Node(fform::Type{F}, variables::SVector{N, Symbol}, factorisation) where { N, F }
    return Node(F, map(v -> nodevar(v), variables), factorisation)
end

functionalform(::Node{F}) where F = F
variables(node::Node)     = node.variables
factorisation(node::Node) = node.factorisation

getcluster(node::Node, i)                = @inbounds node.factorisation[i]
clusters(node::Node)                     = map(factor -> map(i -> node.variables[i], factor), node.factorisation)
clusterindex(node::Node, v::Symbol)      = clusterindex(node, varindex(v))
clusterindex(node::Node, vindex::Int)    = findfirst(cluster -> vindex ∈ cluster, factorisation(node))

varclusterindex(cluster, v::Symbol)   = varclusterindex(cluster, varindex(v))
varclusterindex(cluster, vindex::Int) = findfirst(index -> index === vindex, cluster)

varindex(node::Node, v::Symbol)  = findfirst(d -> d === v, map(v -> name(v), variables(node)))
iscontain(node::Node, v::Symbol) = varindex(node, v) !== nothing
isfactorised(node::Node, f)      = findfirst(d -> d == f, factorisation(node)) !== nothing

function deps(node::Node, v::Symbol)
    vindex = varindex(node, v)
    cindex = clusterindex(node, vindex)

    @assert vindex !== nothing
    @assert cindex !== nothing

    vars = variables(node)
    cls  = factorisation(node)

    factor  = @inbounds cls[cindex]
    vcindex = varclusterindex(factor, vindex)

    @assert vcindex !== nothing

    mdeps       = map(i -> vars[i], skipindex(factor, vcindex))
    clusterdeps = map(i -> vars[i], skipindex(cls, cindex))

    return mdeps, clusterdeps
end
