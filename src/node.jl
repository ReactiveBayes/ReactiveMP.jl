export NodeVariable, nodevar, name, messageout, messagein
export Node, functionalform, variables, factorisation, factors, varindex, iscontain, isfactorised, getvariable
export getcluster, clusters, clusterindex
export deps, connect!, activate!
export rule
export Marginalisation

using StaticArrays
using BenchmarkTools
using Rocket

import Base: show

## Variable constraints

struct Marginalisation end

## Node Variable Props

mutable struct NodeVariableProps
    connected_variable :: Union{Nothing, AbstractVariable}
    connected_index    :: Int

    NodeVariableProps() = new(nothing, 0)
end

## Node Variable

struct NodeVariable
    name  :: Symbol
    m_out :: LazyObservable{AbstractMessage}
    m_in  :: LazyObservable{AbstractMessage}
    props :: NodeVariableProps

    NodeVariable(name::Symbol) = new(name, lazy(AbstractMessage), lazy(AbstractMessage), NodeVariableProps())
end

Base.show(io::IO, nodevar::NodeVariable) = print(io, name(nodevar))

nodevar(name::Symbol) = NodeVariable(name)

name(nodevar::NodeVariable)       = nodevar.name
tag(nodevar::NodeVariable)        = Val(name(nodevar))
messageout(nodevar::NodeVariable) = nodevar.m_out
messagein(nodevar::NodeVariable)  = nodevar.m_in

function connectvariable!(nodevar::NodeVariable, variable, index)
    nodevar.props.connected_variable = variable
    nodevar.props.connected_index    = index
end

connectedvar(nodevar::NodeVariable)      = nodevar.props.connected_variable
connectedvarindex(nodevar::NodeVariable) = nodevar.props.connected_index

## Node

struct Node{F, N, C}
    variables     :: SVector{N, NodeVariable}
    factorisation :: C
end

function Node(::Type{F}, variables::SVector{N, NodeVariable}, factorisation::C) where { F, N, C }
    return Node{F, N, C}(variables, factorisation)
end

function Node(::Type{F}, variables::SVector{N, Symbol}, factorisation) where { N, F }
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

function getvariable(node::Node, v::Symbol)
    vindex = varindex(node, v)
    @assert vindex !== nothing
    return @inbounds variables(node)[vindex]
end

varindex(node::Node, v::Symbol)    = findfirst(d -> d === v, map(v -> name(v), variables(node)))
iscontain(node::Node, v::Symbol)   = varindex(node, v) !== nothing
isfactorised(node::Node, f)        = findfirst(d -> d == f, factorisation(node)) !== nothing

function connect!(node::Node, v::Symbol, variable, index)
    vindex = varindex(node, v)

    @assert vindex !== nothing

    nodevars = variables(node)
    nodevar  = @inbounds nodevars[vindex]

    connectvariable!(nodevar, variable, index)
    setmessagein!(variable, index, messageout(nodevar))
end

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

function activate!(node::Node)
    for variable in variables(node)
        mdeps, clusterdeps = deps(node, name(variable))

        mgsobservable     = length(mdeps) !== 0 ? combineLatest(tuple(map(m -> messagein(m), mdeps)...), true) : of(nothing)
        clusterobservable = length(clusterdeps) !== 0 ? combineLatest(tuple(map(c -> cluster_belief(c), clusterdeps)...), false) : of(nothing)

        fform       = functionalform(node)
        vtag        = tag(variable)
        vconstraint = Marginalisation()
        vmessageout = combineLatest((mgsobservable, clusterobservable), false, (AbstractMessage, (d) -> rule(fform, vtag, vconstraint, d[1], d[2], nothing)))

        set!(messageout(variable), vmessageout |> discontinue() |> share())
        set!(messagein(variable), messageout(connectedvar(variable), connectedvarindex(variable)))
    end
end

function cluster_belief(cluster)
    if length(cluster) === 1 # Cluster contains only one variable, we can take belief over this variable
        connected = connectedvar(cluster[1])
        activate!(connected)
        return getbelief(connected)
    else
        error("Unsupported cluster size: $(length(cluster))")
    end
end

## rule

function rule end

## Helpers for the rule

macro fform(expr)
    return :(::Type{ <: $expr })
end

macro edge(expr)
    return :(::Val{$expr})
end

macro MC()
    return :(::Marginalisation)
end
