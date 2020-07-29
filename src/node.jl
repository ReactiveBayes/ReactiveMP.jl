export VariableNode, varnode, name, messageout, messagein
export FactorNode, functionalform, variables, factorisation, factors, varindex, iscontain, isfactorised, getvariable
export getcluster, clusters, clusterindex
export deps, connect!, activate!
export rule
export Marginalisation

using BenchmarkTools
using Rocket

import Base: show
import Base: getindex, setindex!, firstindex, lastindex

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

## FactorNodeLocalMarginals

struct FactorNodeLocalMarginals{N}
    marginals :: NTuple{N, Tuple{Symbol, Ref{Union{Nothing, LazyObservable{Marginal}}}}}
end

function FactorNodeLocalMarginals(variables, factorisation)
    names = map(n -> Symbol(n...), map(q -> map(v -> variables[v], q), factorisation))
    init  = map(n -> (n, Ref{Union{Nothing, LazyObservable{Marginal}}}(nothing)), names)
    N     = length(factorisation)
    return FactorNodeLocalMarginals{N}(NTuple{N, Tuple{Symbol, Ref{Union{Nothing, LazyObservable{Marginal}}}}}(init))
end

@inline function __findindex(lm::FactorNodeLocalMarginals, s::Symbol)
    index = findnext(d -> d[1] === s, lm.marginals, 1)
    if index === nothing
        throw("Invalid marginal id: $s")
    end
    return index
end

Base.getindex(lm::FactorNodeLocalMarginals, s::Symbol)     = @inbounds lm.marginals[__findindex(lm, s)][2][]
Base.setindex!(lm::FactorNodeLocalMarginals, v, s::Symbol) = @inbounds lm.marginals[__findindex(lm, s)][2][] = v

Base.firstindex(::FactorNodeLocalMarginals)           = 1
Base.lastindex(::FactorNodeLocalMarginals{N}) where N = N

## FactorNode

struct FactorNode{F, N, C, M}
    fform         :: F
    variables     :: NTuple{N, VariableNode}
    factorisation :: C
    marginals     :: M
end

# Additional method specific for Type{F} needed here to bypass Julia's DataType type
function FactorNode(fform::Type{F}, variables::NTuple{N, Symbol}, factorisation::C) where { F, N, C }
    localmarginals = FactorNodeLocalMarginals(variables, factorisation)
    M = typeof(localmarginals)
    return FactorNode{Type{F}, N, C, M}(fform, map(v -> varnode(v), variables), factorisation, localmarginals)
end

function FactorNode(fform::F, variables::NTuple{N, Symbol}, factorisation::C) where { F, N, C }
    localmarginals = FactorNodeLocalMarginals(variables, factorisation)
    M = typeof(localmarginals)
    return FactorNode{F, N, C, M}(fform, map(v -> varnode(v), variables), factorisation, localmarginals)
end

functionalform(factornode::FactorNode) = factornode.fform
variables(factornode::FactorNode)      = factornode.variables
factorisation(factornode::FactorNode)  = factornode.factorisation

getcluster(factornode::FactorNode, i)                = @inbounds factornode.factorisation[i]
clusters(factornode::FactorNode)                     = map(factor -> map(i -> @inbounds factornode.variables[i], factor), factornode.factorisation)
clusterindex(factornode::FactorNode, v::Symbol)      = clusterindex(factornode, varindex(factornode, v))
clusterindex(factornode::FactorNode, vindex::Int)    = findfirst(cluster -> vindex ∈ cluster, factorisation(factornode))

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
        clusterobservable = length(clusterdeps) !== 0 ? combineLatest(map(c -> getmarginal!(factornode, c), clusterdeps)..., strategy = PushEach()) : of(nothing)

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

clustername(cluster) = Symbol(map(v -> name(v), cluster)...)

function setmarginal!(factornode::FactorNode, name::Symbol, marginal)
    # TODO
    throw("Not implemented")
end

function getmarginal!(factornode::FactorNode, cluster)
    cname = clustername(cluster)

    if factornode.marginals[cname] !== nothing
        return factornode.marginals[cname]
    end

    marginal = if length(cluster) === 1 # Cluster contains only one variable, we can take marginal over this variable
        getmarginal(connectedvar(cluster[1]))
    else
        error("Unsupported cluster size: $(length(cluster))")
    end

    factornode.marginals[cname] = marginal

    return marginal
end

## rule

function rule end
