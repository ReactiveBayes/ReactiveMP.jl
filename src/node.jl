export VariableNode, varnode, name, messageout, messagein
export FactorNode, functionalform, variables, factorisation, factors, varindex, iscontain, isfactorised, getvariable
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

## FactorNode Variable Props

mutable struct VariableNodeProps
    connected_variable :: Union{Nothing, AbstractVariable}
    connected_index    :: Int

    VariableNodeProps() = new(nothing, 0)
end

## FactorNode Variable

struct VariableNode
    name  :: Symbol
    m_out :: LazyObservable{Message}
    m_in  :: LazyObservable{Message}
    props :: VariableNodeProps

    VariableNode(name::Symbol) = new(name, lazy(Message), lazy(Message), VariableNodeProps())
end

Base.show(io::IO, varnode::VariableNode) = print(io, name(varnode))

varnode(name::Symbol) = VariableNode(name)

name(varnode::VariableNode)       = varnode.name
tag(varnode::VariableNode)        = Val(name(varnode))
messageout(varnode::VariableNode) = varnode.m_out
messagein(varnode::VariableNode)  = varnode.m_in

function connectvariable!(varnode::VariableNode, variable, index)
    varnode.props.connected_variable = variable
    varnode.props.connected_index    = index
end

connectedvar(varnode::VariableNode)      = varnode.props.connected_variable
connectedvarindex(varnode::VariableNode) = varnode.props.connected_index

## FactorNode
# TODO: posterior_factorisation

struct FactorNode{F, N, C}
    fform         :: F
    variables     :: SVector{N, VariableNode}
    factorisation :: C
end

function FactorNode(fform::Type{F}, variables::SVector{N, Symbol}, factorisation::C) where { N, F, C }
    return FactorNode{Type{F}, N, C}(fform, map(v -> varnode(v), variables), factorisation)
end

function FactorNode(fform::F, variables::SVector{N, Symbol}, factorisation::C) where { N, F, C }
    return FactorNode{F, N, C}(fform, map(v -> varnode(v), variables), factorisation)
end

functionalform(node::FactorNode) = node.fform
variables(node::FactorNode)      = node.variables
factorisation(node::FactorNode)  = node.factorisation

getcluster(node::FactorNode, i)                = @inbounds node.factorisation[i]
clusters(node::FactorNode)                     = map(factor -> map(i -> @inbounds node.variables[i], factor), node.factorisation)
clusterindex(node::FactorNode, v::Symbol)      = clusterindex(node, varindex(v))
clusterindex(node::FactorNode, vindex::Int)    = findfirst(cluster -> vindex ∈ cluster, factorisation(node))

varclusterindex(cluster, v::Symbol)   = varclusterindex(cluster, varindex(v))
varclusterindex(cluster, vindex::Int) = findfirst(index -> index === vindex, cluster)

function getvariable(node::FactorNode, v::Symbol)
    vindex = varindex(node, v)
    @assert vindex !== nothing
    return @inbounds variables(node)[vindex]
end

varindex(node::FactorNode, v::Symbol)    = findfirst(d -> d === v, map(v -> name(v), variables(node)))
iscontain(node::FactorNode, v::Symbol)   = varindex(node, v) !== nothing
isfactorised(node::FactorNode, f)        = findfirst(d -> d == f, factorisation(node)) !== nothing

function connect!(node::FactorNode, v::Symbol, variable, index)
    vindex = varindex(node, v)

    @assert vindex !== nothing

    varnodes = variables(node)
    varnode  = @inbounds varnodes[vindex]

    connectvariable!(varnode, variable, index)
    setmessagein!(variable, index, messageout(varnode))
end

function deps(node::FactorNode, v::Symbol)
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

function activate!(model, node::FactorNode)
    for variable in variables(node)
        mdeps, clusterdeps = deps(node, name(variable))

        mgsobservable     = length(mdeps) !== 0 ? combineLatest(map(m -> messagein(m), mdeps)..., strategy = PushNew()) : of(nothing)
        clusterobservable = length(clusterdeps) !== 0 ? combineLatest(map(c -> cluster_marginal(c), clusterdeps)..., strategy = PushEach()) : of(nothing)

        gate        = message_gate(model)
        fform       = functionalform(node)
        vtag        = tag(variable)
        vconstraint = Marginalisation()
        mapping     = map(Message, (d) -> as_message(gate!(gate, node, variable, rule(fform, vtag, vconstraint, d[1], d[2], nothing))))
        vmessageout = combineLatest(mgsobservable, clusterobservable, strategy = PushEach()) |> mapping

        set!(messageout(variable), vmessageout |> discontinue() |> share())
        set!(messagein(variable), messageout(connectedvar(variable), connectedvarindex(variable)))
    end
end

function cluster_marginal(cluster)
    if length(cluster) === 1 # Cluster contains only one variable, we can take marginal over this variable
        return getmarginal(connectedvar(cluster[1]))
    else
        error("Unsupported cluster size: $(length(cluster))")
    end
end

## rule

function rule end
