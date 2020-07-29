export VariableNode, varnode, name, messageout, messagein
export FactorNode, functionalform, variables, factorisation, factors, varindex, iscontain, isfactorised, getvariable
export getcluster, clusters, clusterindex
export deps, connect!, activate!
export rule
export Marginalisation

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

struct FactorNode{F, N, C}
    fform         :: F
    variables     :: NTuple{N, VariableNode}
    factorisation :: C
end

# Additional method specific for Type{F} needed here to bypass Julia's DataType type
function FactorNode(fform::Type{F}, variables::NTuple{N, Symbol}, factorisation::C) where { F, N, C }
    return FactorNode{Type{F}, N, C}(fform, map(v -> varnode(v), variables), factorisation)
end

function FactorNode(fform::F, variables::NTuple{N, Symbol}, factorisation::C) where { F, N, C }
    return FactorNode{F, N, C}(fform, map(v -> varnode(v), variables), factorisation)
end

functionalform(factornode::FactorNode) = factornode.fform
variables(factornode::FactorNode)      = factornode.variables
factorisation(factornode::FactorNode)  = factornode.factorisation

getcluster(factornode::FactorNode, i)                = @inbounds factornode.factorisation[i]
clusters(factornode::FactorNode)                     = map(factor -> map(i -> @inbounds factornode.variables[i], factor), factornode.factorisation)
clusterindex(factornode::FactorNode, v::Symbol)      = clusterindex(factornode, varindex(v))
clusterindex(factornode::FactorNode, vindex::Int)    = findfirst(cluster -> vindex ∈ cluster, factorisation(factornode))

varclusterindex(cluster, v::Symbol)   = varclusterindex(cluster, varindex(v))
varclusterindex(cluster, vindex::Int) = findfirst(index -> index === vindex, cluster)

function getvariable(factornode::FactorNode, v::Symbol)
    vindex = varindex(factornode, v)
    @assert vindex !== nothing
    return @inbounds variables(factornode)[vindex]
end

varindex(factornode::FactorNode, v::Symbol)    = findfirst(d -> d === v, map(v -> name(v), variables(factornode)))
iscontain(factornode::FactorNode, v::Symbol)   = varindex(factornode, v) !== nothing
isfactorised(factornode::FactorNode, f)        = findfirst(d -> d == f, factorisation(factornode)) !== nothing

function connect!(factornode::FactorNode, v::Symbol, variable, index)
    vindex = varindex(factornode, v)

    @assert vindex !== nothing

    varnodes = variables(factornode)
    varnode  = @inbounds varnodes[vindex]

    connectvariable!(varnode, variable, index)
    setmessagein!(variable, index, messageout(varnode))
end

function deps(factornode::FactorNode, v::Symbol)
    vindex = varindex(factornode, v)
    cindex = clusterindex(factornode, vindex)

    @assert vindex !== nothing
    @assert cindex !== nothing

    vars = variables(factornode)
    cls  = factorisation(factornode)

    factor  = @inbounds cls[cindex]
    vcindex = varclusterindex(factor, vindex)

    @assert vcindex !== nothing

    mdeps       = map(i -> vars[i], skipindex(factor, vcindex))
    clusterdeps = map(i -> vars[i], skipindex(cls, cindex))

    return mdeps, clusterdeps
end

function activate!(model, factornode::FactorNode)
    for variable in variables(factornode)
        mdeps, clusterdeps = deps(factornode, name(variable))

        mgsobservable     = length(mdeps) !== 0 ? combineLatest(map(m -> messagein(m), mdeps)..., strategy = PushNew()) : of(nothing)
        clusterobservable = length(clusterdeps) !== 0 ? combineLatest(map(c -> cluster_marginal(c), clusterdeps)..., strategy = PushEach()) : of(nothing)

        gate        = message_gate(model)
        fform       = functionalform(factornode)
        vtag        = tag(variable)
        vconstraint = Marginalisation()
        mapping     = map(Message, (d) -> as_message(gate!(gate, factornode, variable, rule(fform, vtag, vconstraint, d[1], d[2], nothing))))
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
