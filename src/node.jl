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

## VariableNode Props

mutable struct VariableNodeProps
    connected_variable :: Union{Nothing, AbstractVariable}
    connected_index    :: Int

    VariableNodeProps() = new(nothing, 0)
end

## VariableNode

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
    marginals :: NTuple{ N, Tuple{ Symbol, Ref{ Union{ Nothing, MarginalObservable } } } }
end

function FactorNodeLocalMarginals(variables, factorisation)
    names = clusternames(variables, factorisation)
    init  = map(n -> (n, Ref{Union{Nothing, MarginalObservable}}(nothing)), names)
    N     = length(factorisation)
    return FactorNodeLocalMarginals{N}(NTuple{N, Tuple{Symbol, Ref{Union{Nothing, MarginalObservable}}}}(init))
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

clusternames(factornode::FactorNode)                              = clusternames(map(v -> name(v), variables(factornode)), factorisation(factornode))
clusternames(variables::NTuple{N, Symbol}, factorisation) where N = map(n -> Symbol(n...), map(q -> map(v -> variables[v], q), factorisation))

getcluster(factornode::FactorNode, i)                = @inbounds factornode.factorisation[i]
clusters(factornode::FactorNode)                     = map(factor -> map(i -> @inbounds factornode.variables[i], factor), factornode.factorisation)
clusterindex(factornode::FactorNode, v::Symbol)      = clusterindex(factornode, varindex(factornode, v))
clusterindex(factornode::FactorNode, vindex::Int)    = findfirst(cluster -> vindex ∈ cluster, factorisation(factornode))

clusterindex(factornode::FactorNode, vars::NTuple{N, VariableNode}) where N = clusterindex(factornode, map(v -> name(v), vars))
clusterindex(factornode::FactorNode, vars::NTuple{N, Symbol}) where N       = clusterindex(factornode, map(v -> varindex(factornode, v), vars))
clusterindex(factornode::FactorNode, vars::NTuple{N, Int}) where N          = findfirst(cluster -> all(v -> v ∈ cluster, vars), factorisation(factornode))

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

    mdeps       = map(inds -> map(i -> vars[i], inds), skipindex(factor, vcindex))
    clusterdeps = map(inds -> map(i -> vars[i], inds), skipindex(cls, cindex))

    return mdeps, clusterdeps
end

function activate!(model, factornode::FactorNode)
    for variable in variables(factornode)
        mdeps, clusterdeps = deps(factornode, name(variable))

        msgs_observable     = length(mdeps)       !== 0 ? combineLatest(map(m -> messagein(m), mdeps)..., strategy = PushNew()) : of(nothing)
        clusters_observable = length(clusterdeps) !== 0 ? combineLatest(map(c -> getmarginal!(factornode, c), clusterdeps)..., strategy = PushEach()) : of(nothing)

        gate        = message_gate(model)
        fform       = functionalform(factornode)
        vtag        = tag(variable)
        vconstraint = Marginalisation()
        mapping     = map(Message, (d) -> as_message(gate!(gate, factornode, variable, rule(fform, vtag, vconstraint, d[1], d[2], nothing))))
        vmessageout = combineLatest(msgs_observable, clusters_observable, strategy = PushEach()) |> discontinue() |> mapping

        set!(messageout(variable), vmessageout |> share())
        set!(messagein(variable), messageout(connectedvar(variable), connectedvarindex(variable)))
    end
end

clustername(cluster) = Symbol(map(v -> name(v), cluster)...)

function setmarginal!(factornode::FactorNode, name::Symbol, v)
    marginal = factornode.marginals[name]
    if marginal === nothing
        throw("Marginal with name $(name) does not exist on factor node $(factornode)")
    end
    setmarginal!(marginal, v)
end

function getmarginal!(factornode::FactorNode, cluster)
    cname = clustername(cluster)

    if factornode.marginals[cname] !== nothing
        return factornode.marginals[cname]
    end

    if length(cluster) === 1 # Cluster contains only one variable, we can take marginal over this variable
        vmarginal = getmarginal(connectedvar(cluster[1]))
        factornode.marginals[cname] = vmarginal
        return vmarginal
    else
        cmarginal = MarginalObservable()
        factornode.marginals[cname] = cmarginal
        # TODO generalise as a separate function
        mdeps = cluster

        vars = variables(factornode)
        cls  = factorisation(factornode)

        cindex      = clusterindex(factornode, cluster)
        clusterdeps = map(inds -> map(i -> vars[i], inds), skipindex(cls, cindex))

        msgs_observable     = length(mdeps)       !== 0 ? combineLatest(map(m -> messagein(m), mdeps)..., strategy = PushNew()) : of(nothing)
        clusters_observable = length(clusterdeps) !== 0 ? combineLatest(map(c -> getmarginal!(factornode, c), clusterdeps)..., strategy = PushEach()) : of(nothing)

        fform       = functionalform(factornode)
        vtag        = Val(clustername(cluster))
        mapping     = map(Marginal, (d) -> as_marginal(marginalrule(fform, vtag, d[1], d[2], nothing)))
        marginalout = combineLatest(msgs_observable, clusters_observable, strategy = PushEach()) |> discontinue() |> mapping

        connect!(cmarginal, marginalout)

        return cmarginal
    end

    throw("Unsupported marginal size: $(length(cluster))")
end

## rule

function rule end
