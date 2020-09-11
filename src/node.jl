export VariableNode, varnode, name, messageout, messagein
export FactorNode, functionalform, variables, factorisation, factors, varindex, iscontain, isfactorised, getvariable
export getcluster, clusters, clusterindex
export deps, connect!, activate!
export rule
export make_node
export Marginalisation
export sdtype, isdeterministic, isstochastic

using BenchmarkTools
using Rocket

import Base: show
import Base: getindex, setindex!, firstindex, lastindex

## Node types

struct Deterministic end
struct Stochastic end

isdeterministic(::Deterministic)         = true
isdeterministic(::Type{ Deterministic }) = true
isdeterministic(::Stochastic)            = false
isdeterministic(::Type{ Stochastic })    = false

isstochastic(::Stochastic)            = true
isstochastic(::Type{ Stochastic })    = true
isstochastic(::Deterministic)         = false
isstochastic(::Type{ Deterministic }) = false

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

name(symbol::Symbol)              = symbol
name(varnode::VariableNode)       = name(varnode.name)
tag(varnode::VariableNode)        = Val{name(varnode)}
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

struct FactorNode{F, T, N, C, M, A}
    fform         :: F
    variables     :: NTuple{N, VariableNode}
    factorisation :: C
    marginals     :: M
    metadata      :: A
end

# Additional method specific for Type{F} needed here to bypass Julia's DataType type
function FactorNode(fform::Type{F}, ::Type{T}, variables::NTuple{N, Symbol}, factorisation::C, metadata::A) where { F, N, C, T, A }
    localmarginals = FactorNodeLocalMarginals(variables, factorisation)
    M = typeof(localmarginals)
    return FactorNode{Type{F}, T, N, C, M, A}(fform, map(v -> varnode(v), variables), factorisation, localmarginals, metadata)
end

function FactorNode(fform::F, ::Type{T}, variables::NTuple{N, Symbol}, factorisation::C, metadata::A) where { F, N, C, T, A }
    localmarginals = FactorNodeLocalMarginals(variables, factorisation)
    M = typeof(localmarginals)
    return FactorNode{F, T, N, C, M, A}(fform, map(v -> varnode(v), variables), factorisation, localmarginals, metadata)
end

functionalform(factornode::FactorNode) = factornode.fform
variables(factornode::FactorNode)      = factornode.variables
factorisation(factornode::FactorNode)  = factornode.factorisation

sdtype(::N) where { N <: FactorNode } = sdtype(N)
sdtype(::Type{ <: FactorNode{F, T} }) where { F, T } = T

isstochastic(::N)    where { N <: FactorNode }         = isstochastic(N)
isstochastic(::Type{ N })    where { N <: FactorNode } = isstochastic(sdtype(N))

isdeterministic(::N) where { N <: FactorNode }         = isdeterministic(N)
isdeterministic(::Type{ N }) where { N <: FactorNode } = isdeterministic(sdtype(N))

metadata(factornode::FactorNode) = factornode.metadata

clustername(cluster) = mapreduce(v -> name(v), (a, b) -> Symbol(a, :_, b), cluster)

clusternames(factornode::FactorNode)                              = map(clustername, clusters(factornode))
clusternames(variables::NTuple{N, Symbol}, factorisation) where N = map(clustername, map(q -> map(v -> variables[v], q), factorisation))

getcluster(factornode::FactorNode, i)                = @inbounds factornode.factorisation[i]
clusters(factornode::FactorNode)                     = map(factor -> map(i -> begin return @inbounds factornode.variables[i] end, factor), factorisation(factornode))
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

function connect!(factornode::FactorNode, v::Symbol, variable) 
    return connect!(factornode::FactorNode, v::Symbol, variable, getlastindex(variable))
end

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

    # TODO Consider to change this line with map/map
    # TODO benchmark it
    mdeps       = map(inds -> map(i -> begin return @inbounds vars[i] end, inds), skipindex(factor, vcindex))
    clusterdeps = map(inds -> map(i -> begin return @inbounds vars[i] end, inds), skipindex(cls, cindex))

    return mdeps, clusterdeps
end

function activate!(model, factornode::FactorNode)
    for variable in variables(factornode)
        mdeps, clusterdeps = deps(factornode, name(variable))

        msgs_names      = nothing
        msgs_observable = of(nothing)

        cluster_names       = nothing
        clusters_observable = of(nothing)

        if length(mdeps) !== 0
            msgs_names      = Val{ tuple(name.(mdeps)...) }
            msgs_observable = combineLatest(map(m -> messagein(m), mdeps)..., strategy = PushNew())
        end

        if length(clusterdeps) !== 0 
            cluster_names       = Val{ tuple(clustername.(clusterdeps)...) }
            clusters_observable = combineLatest(map(c -> getmarginal!(factornode, c), clusterdeps)..., strategy = PushEach())
        end

        gate        = message_gate(model)
        fform       = functionalform(factornode)
        vtag        = tag(variable)
        vconstraint = Marginalisation()
        meta        = metadata(factornode)
        mapping     = map(Message, (d) -> as_message(gate!(gate, factornode, variable, rule(fform, vtag, vconstraint, msgs_names, d[1], cluster_names, d[2], meta))))
        vmessageout = combineLatest(msgs_observable, clusters_observable, strategy = PushEach()) |> discontinue() |> mapping

        set!(messageout(variable), vmessageout |> share_replay(1))
        set!(messagein(variable), messageout(connectedvar(variable), connectedvarindex(variable)))
    end
end

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

        msgs_names      = nothing
        msgs_observable = of(nothing)

        cluster_names       = nothing
        clusters_observable = of(nothing)

        if length(mdeps) !== 0
            msgs_names      = Val{ tuple(name.(mdeps)...) }
            msgs_observable = combineLatest(map(m -> messagein(m), mdeps)..., strategy = PushNew())
        end

        if length(clusterdeps) !== 0 
            cluster_names       = Val{ tuple(clustername.(clusterdeps)...) }
            clusters_observable = combineLatest(map(c -> getmarginal!(factornode, c), clusterdeps)..., strategy = PushEach())
        end

        fform       = functionalform(factornode)
        vtag        = Val{ clustername(cluster) }
        meta        = metadata(factornode)
        mapping     = map(Marginal, (d) -> as_marginal(marginalrule(fform, vtag, nothing, msgs_names, d[1], cluster_names, d[2], meta)))
        marginalout = combineLatest(msgs_observable, clusters_observable, strategy = PushEach()) |> discontinue() |> mapping

        connect!(cmarginal, marginalout |> share_replay(1))

        return cmarginal
    end

    throw("Unsupported marginal size: $(length(cluster))")
end

## rule

function rule end
function marginalrule end

function make_node end
