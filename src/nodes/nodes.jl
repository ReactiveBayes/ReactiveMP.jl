export Deterministic, Stochastic, isdeterministic, isstochastic, sdtype
export MeanField, FullFactorisation, Marginalisation, MomentMatching
export functionalform, interfaces, factorisation, localmarginals, localmarginalnames, metadata
export FactorNodesCollection, getnodes, getnode_ids
export make_node, FactorNodeCreationOptions
export GenericFactorNode
export @node

using Rocket
using TupleTools
using MacroTools

import Rocket: getscheduler

import Base: show, +, push!, iterate, IteratorSize, IteratorEltype, eltype, length, size
import Base: getindex, setindex!, firstindex, lastindex

## Node traits

"""
    ValidNodeFunctionalForm

Trait specification for an object that can be used in model specification as a factor node.

See also: [`ReactiveMP.as_node_functional_form`](@ref), [`ReactiveMP.UndefinedNodeFunctionalForm`](@ref)
"""
struct ValidNodeFunctionalForm end

"""
    UndefinedNodeFunctionalForm

Trait specification for an object that can **not** be used in model specification as a factor node.

See also: [`ReactiveMP.as_node_functional_form`](@ref), [`ReactiveMP.ValidNodeFunctionalForm`](@ref)
"""
struct UndefinedNodeFunctionalForm end

"""
    as_node_functional_form(object)

Determines `object` node functional form trait specification.
Returns either `ValidNodeFunctionalForm()` or `UndefinedNodeFunctionalForm()`.

See also: [`ReactiveMP.ValidNodeFunctionalForm`](@ref), [`ReactiveMP.UndefinedNodeFunctionalForm`](@ref)
"""
function as_node_functional_form end

as_node_functional_form(some) = UndefinedNodeFunctionalForm()

## Node types

"""
    Deterministic

`Deterministic` object used to parametrize factor node object with determinstic type of relationship between variables.

See also: [`Stochastic`](@ref), [`isdeterministic`](@ref), [`isstochastic`](@ref), [`sdtype`](@ref)
"""
struct Deterministic end

"""
    Stochastic

`Stochastic` object used to parametrize factor node object with stochastic type of relationship between variables.

See also: [`Deterministic`](@ref), [`isdeterministic`](@ref), [`isstochastic`](@ref), [`sdtype`](@ref)
"""
struct Stochastic end

"""
    isdeterministic(node)

Function used to check if factor node object is deterministic or not. Returns true or false.

See also: [`Deterministic`](@ref), [`Stochastic`](@ref), [`isstochastic`](@ref), [`sdtype`](@ref)
"""
function isdeterministic end

"""
    isstochastic(node)

Function used to check if factor node object is stochastic or not. Returns true or false.

See also: [`Deterministic`](@ref), [`Stochastic`](@ref), [`isdeterministic`](@ref), [`sdtype`](@ref)
"""
function isstochastic end

isdeterministic(::Deterministic)       = true
isdeterministic(::Type{Deterministic}) = true
isdeterministic(::Stochastic)          = false
isdeterministic(::Type{Stochastic})    = false

isstochastic(::Stochastic)          = true
isstochastic(::Type{Stochastic})    = true
isstochastic(::Deterministic)       = false
isstochastic(::Type{Deterministic}) = false

"""
    sdtype(object)

Returns either `Deterministic` or `Stochastic` for a given object (if defined).

See also: [`Deterministic`](@ref), [`Stochastic`](@ref), [`isdeterministic`](@ref), [`isstochastic`](@ref)
"""
function sdtype end

# TODO: bvdmitri remove this
# Any `Type` is considered to be a deterministic mapping unless stated otherwise (By convention, any `Distribution` type is not deterministic)
# E.g. `Matrix` is not an instance of the `Function` abstract type, however we would like to pretend it is a deterministic function
sdtype(::Type{T}) where {T}    = Deterministic()
sdtype(::Type{<:Distribution}) = Stochastic()
sdtype(::Function)             = Deterministic()

"""
    as_node_symbol(type)

Returns a symbol associated with a node `type`.
"""
function as_node_symbol end

as_node_symbol(fn::F) where {F <: Function} = Symbol(fn)

## Generic factorisation constraints

"""
    MeanField

Generic factorisation constraint used to specify a mean-field factorisation for recognition distribution `q`.

See also: [`FullFactorisation`](@ref)
"""
struct MeanField end

"""
    FullFactorisation

Generic factorisation constraint used to specify a full factorisation for recognition distribution `q`.

See also: [`MeanField`](@ref)
"""
struct FullFactorisation end

"""
    collect_factorisation(nodetype, factorisation)

This function converts given factorisation to a correct internal factorisation representation for a given node.

See also: [`MeanField`](@ref), [`FullFactorisation`](@ref)
"""
function collect_factorisation end

"""
    collect_meta(nodetype, meta)

This function converts given meta object to a correct internal meta representation for a given node.
Fallbacks to `default_meta` in case if meta is `nothing`.

See also: [`default_meta`](@ref), [`FactorNode`](@ref)
"""
function collect_meta end

collect_meta(T::Any, ::Nothing) = default_meta(T)
collect_meta(T::Any, meta::Any) = meta

"""
    default_meta(nodetype)

Returns default meta object for a given node type.

See also: [`collect_meta`](@ref), [`FactorNode`](@ref)
"""
function default_meta end

default_meta(any) = nothing

## NodeInterface

struct Marginalisation end
struct MomentMatching end

include("interfaces.jl")
include("clusters.jl")

abstract type AbstractFactorNode end

## Generic Factor node new code start

"""
    GenericFactorNode(functionalform, interfaces)

Generic factor node object that represents a factor node with a given `functionalform` and `interfaces`.
"""
struct GenericFactorNode{F, I} <: AbstractFactorNode
    interfaces::I

    function GenericFactorNode(::Type{F}, interfaces::I) where {F, I <: Tuple}
        return new{F, I}(interfaces)
    end
end

GenericFactorNode(::Type{F}, interfaces::I) where {F, I} = GenericFactorNode(F, __prepare_interfaces_generic(interfaces))
GenericFactorNode(::F, interfaces::I) where {F <: Function, I} = GenericFactorNode(F, __prepare_interfaces_generic(interfaces))

functionalform(factornode::GenericFactorNode{F}) where {F} = F
getinterfaces(factornode::GenericFactorNode) = factornode.interfaces
getinterface(factornode::GenericFactorNode, index) = factornode.interfaces[index]

# Takes a named tuple of abstract variables and converts to a tuple of NodeInterfaces with the same order
function __prepare_interfaces_generic(interfaces::NamedTuple)
    return map(key -> NodeInterface(key, interfaces[key]), keys(interfaces))
end

## activate!

struct FactorNodeActivationOptions{C, M, D, P, A, S}
    factorization::C
    metadata::M
    dependencies::D
    pipeline::P
    addons::A
    scheduler::S
end

getfactorization(options::FactorNodeActivationOptions) = options.factorization
getmetadata(options::FactorNodeActivationOptions) = options.metadata
getdependecies(options::FactorNodeActivationOptions) = options.dependencies
getpipeline(options::FactorNodeActivationOptions) = options.pipeline
getaddons(options::FactorNodeActivationOptions) = options.addons
getscheduler(options::FactorNodeActivationOptions) = options.scheduler

function activate!(factornode::GenericFactorNode, options::FactorNodeActivationOptions)
    scheduler     = getscheduler(options)
    addons        = getaddons(options)
    fform         = functionalform(factornode)
    factorization = collect_factorisation(fform, getfactorization(options))
    meta          = collect_meta(fform, getmetadata(options))
    dependencies  = collect_functional_dependencies(fform, getdependecies(options))
    pipeline      = collect_pipeline(fform, getpipeline(options))
    clusters      = FactorNodeLocalClusters(factornode.interfaces, factorization)

    activate!(factornode, options, clusters)

    foreach(enumerate(getinterfaces(factornode))) do (iindex, interface)
        if israndom(interface) || isdata(interface)
            message_dependencies, marginal_dependencies = functional_dependencies(dependencies, factornode, clusters, interface, iindex)

            msgs_names, msgs_observable          = get_messages_observable(factornode, message_dependencies)
            marginal_names, marginals_observable = get_marginals_observable(factornode, marginal_dependencies)

            vtag        = tag(interface)
            vconstraint = Marginalisation()

            vmessageout = combineLatest((msgs_observable, marginals_observable), PushNew())

            mapping = let messagemap = MessageMapping(fform, vtag, vconstraint, msgs_names, marginal_names, meta, addons, node_if_required(fform, factornode))
                (dependencies) -> VariationalMessage(dependencies[1], dependencies[2], messagemap)
            end

            vmessageout = vmessageout |> map(AbstractMessage, mapping)
            vmessageout = apply_pipeline_stage(pipeline, factornode, vtag, vmessageout)
            vmessageout = vmessageout |> schedule_on(scheduler)

            connect!(messageout(interface), vmessageout)
        end
    end
end

function activate!(factornode::GenericFactorNode, options::FactorNodeActivationOptions, clusters::FactorNodeLocalClusters)
    # We first preinitialize the `MarginalObservable` for those clusters, which length is not equal to `1`
    # For the clusters which length is equal to one we simple reuse the marginal of the connected variable
    foreach(enumerate(getfactorization(clusters))) do (index, localfactorization)
        cmarginal = if isone(length(localfactorization))
            getmarginal(getvariable(getinterface(factornode, first(localfactorization))), IncludeAll())
        else
            MarginalObservable()
        end
        setstream!(getmarginal(clusters, index), cmarginal)
    end
    # After all streams have been initialized we can create streams that are needed to compute them
    foreach(enumerate(getfactorization(clusters))) do (index, localfactorization)
        if !isone(length(localfactorization))
            activate!(factornode, options, localfactorization, clusters, getmarginal(clusters, index), index)
        end
    end
end

function activate!(
    factornode::GenericFactorNode,
    options::FactorNodeActivationOptions,
    localfactorization::Tuple,
    clusters::FactorNodeLocalClusters,
    localmarginal::FactorNodeLocalMarginal,
    index::Int
)
    if !isone(length(localfactorization))
        cmarginal = getstream(localmarginal)

        clusterinterfaces = map(i -> getinterface(factornode, i), localfactorization)

        message_dependencies  = tuple(clusterinterfaces...)
        marginal_dependencies = tuple(TupleTools.deleteat(getmarginals(clusters), index)...)

        msgs_names, msgs_observable          = get_messages_observable(factornode, message_dependencies)
        marginal_names, marginals_observable = get_marginals_observable(factornode, marginal_dependencies)

        fform = functionalform(factornode)
        vtag  = tag(localmarginal)
        meta  = collect_meta(fform, getmetadata(options))

        mapping = MarginalMapping(fform, vtag, msgs_names, marginal_names, meta, node_if_required(fform, factornode))
        # TODO: discontinue operator is needed for loopy belief propagation? Check
        marginalout = combineLatest((msgs_observable, marginals_observable), PushNew()) |> discontinue() |> map(Marginal, mapping)

        connect!(cmarginal, marginalout)
    else
        @warn "`activate!` should not be called for local cluster of length `1`."
    end
end

## Generic Factor Node new code end

struct FactorNodeCreationOptions{F, M, P}
    factorisation :: F
    metadata      :: M
    pipeline      :: P
end

# FactorNodeCreationOptions() = FactorNodeCreationOptions(nothing, nothing, nothing)

# factorisation(options::FactorNodeCreationOptions) = options.factorisation
# metadata(options::FactorNodeCreationOptions)      = options.metadata
# getpipeline(options::FactorNodeCreationOptions)   = options.pipeline

# Base.broadcastable(options::FactorNodeCreationOptions) = Ref(options)

# Removed struct
struct FactorNodesCollection end

struct FactorNode{F, I, C, M, A, P} <: AbstractFactorNode
    fform          :: F
    interfaces     :: I
    factorisation  :: C
    localmarginals :: M
    metadata       :: A
    pipeline       :: P
end

# function FactorNode(fform::Type{F}, interfaces::I, factorisation::C, localmarginals::M, metadata::A, pipeline::P) where {F, I, C, M, A, P}
#     return FactorNode{Type{F}, I, C, M, A, P}(fform, interfaces, factorisation, localmarginals, metadata, pipeline)
# end

# function FactorNode(fform, varnames::NTuple{N, Symbol}, factorisation, metadata, pipeline) where {N}
#     interfaces     = map(varname -> NodeInterface(varname, interface_default_local_constraint(fform, varname)), varnames)
#     localmarginals = FactorNodeLocalClusters(varnames, factorisation)
#     return FactorNode(fform, interfaces, factorisation, localmarginals, metadata, pipeline)
# end

# function Base.show(io::IO, factornode::FactorNode)
#     println(io, "FactorNode:")
#     println(io, string(" form            : ", functionalform(factornode)))
#     println(io, string(" sdtype          : ", sdtype(factornode)))
#     println(io, string(" interfaces      : ", interfaces(factornode)))
#     println(io, string(" factorisation   : ", factorisation(factornode)))
#     println(io, string(" local marginals : ", localmarginalnames(factornode)))
#     println(io, string(" metadata        : ", metadata(factornode)))
#     println(io, string(" pipeline        : ", getpipeline(factornode)))
# end

# functionalform(factornode::FactorNode)     = factornode.fform
# sdtype(factornode::FactorNode)             = sdtype(functionalform(factornode))
# interfaces(factornode::FactorNode)         = factornode.interfaces
# connectedvars(factornode::FactorNode)      = map((i) -> connected_properties(i), interfaces(factornode))
# factorisation(factornode::FactorNode)      = factornode.factorisation
# localmarginals(factornode::FactorNode)     = factornode.localmarginals.marginals
# localmarginalnames(factornode::FactorNode) = map(name, localmarginals(factornode))
# metadata(factornode::FactorNode)           = factornode.metadata
# getpipeline(factornode::FactorNode)        = factornode.pipeline

# function nodefunction(factornode::FactorNode)
#     return let fform = functionalform(factornode)
#         (out, inputs...) -> pdf(convert(fform, inputs...), out)
#     end
# end

# clustername(cluster) = mapreduce(v -> name(v), (a, b) -> Symbol(a, :_, b), cluster)

# # Cluster is reffered to a tuple of node interfaces
# clusters(factornode::FactorNode) = map(factor -> map(i -> @inbounds(interfaces(factornode)[i]), factor), factorisation(factornode))

# clusterindex(factornode::FactorNode, v::Symbol)                                = clusterindex(factornode, (v,))
# clusterindex(factornode::FactorNode, vindex::Int)                              = clusterindex(factornode, (vindex,))
# clusterindex(factornode::FactorNode, vars::NTuple{N, NodeInterface}) where {N} = clusterindex(factornode, map(v -> name(v), vars))
# clusterindex(factornode::FactorNode, vars::NTuple{N, Symbol}) where {N}        = clusterindex(factornode, map(v -> interfaceindex(factornode, v), vars))

# # We assume that `inbound` interfaces have indices 2..N
# inboundinterfaces(factornode::FactorNode)  = TupleTools.deleteat(interfaces(factornode), 1)
# inboundclustername(factornode::FactorNode) = clustername(inboundinterfaces(factornode))

# function interfaceindex(factornode::FactorNode, iname::Symbol)
#     iindex = findfirst(interface -> name(interface) === iname, interfaces(factornode))
#     return iindex !== nothing ? iindex : error("Unknown interface ':$(iname)' for $(functionalform(factornode)) node")
# end

# function clusterindex(factornode::FactorNode, vars::NTuple{N, Int}) where {N}
#     cindex = findfirst(cluster -> all(v -> v ∈ cluster, vars), factorisation(factornode))
#     return cindex !== nothing ? cindex : error("Unknown cluster '$(vars)' for $(functionalform(factornode)) node")
# end

# function varclusterindex(cluster, iindex::Int)
#     vcindex = findfirst(index -> index === iindex, cluster)
#     return vcindex !== nothing ? vcindex : error("Invalid cluster '$(vars)' for a given interface index '$(iindex)'")
# end

# getinterface(factornode::FactorNode, iname::Symbol) = @inbounds interfaces(factornode)[interfaceindex(factornode, iname)]

# getclusterinterfaces(factornode::FactorNode, cindex::Int) = @inbounds map(i -> interfaces(factornode)[i], factorisation(factornode)[cindex])

# iscontain(factornode::FactorNode, iname::Symbol) = findfirst(interface -> name(interface) === iname, interfaces(factornode)) !== nothing
# isfactorised(factornode::FactorNode, factor)     = findfirst(f -> f == factor, factorisation(factornode)) !== nothing

## Node pipeline

include("dependencies.jl")

function make_node end # TODO (wouterwln) remove this, but it breaks precompilation because of node definitions downstream

## macro helpers

import .MacroHelpers

function correct_interfaces end

alias_group(s::Symbol) = [s]
function alias_group(e::Expr)
    if @capture(e, (s_, aliases = aliases_))
        result = [s, aliases.args...]
        if length(result) != length(unique(result))
            error("Aliases should be unique")
        end
        return result
    else
        return [e]
    end
end

check_all_symbol(::AbstractArray{T} where {T <: NTuple{N, Symbol} where {N}}) = nothing
check_all_symbol(::Any) = error("All interfaces should be symbols")

macro node(node_fform, node_type, node_interfaces, interface_aliases)
    # Assert that the node type is either Stochastic or Deterministic, and that all interfaces are symbols
    @assert node_type ∈ [:Stochastic, :Deterministic]
    @assert length(node_interfaces.args) > 0
    
    interface_alias_groups = map(alias_group, node_interfaces.args)
    all_aliases = vec(collect(Iterators.product(interface_alias_groups...)))



    # Determine whether we should dispatch on `typeof($fform)` or `Type{$node_fform}`
    if @capture(node_fform, typeof(fform_))
        dispatch_type = quote typeof($fform) end
    else

        dispatch_type = quote Type{$node_fform} end
    end
    # Define the necessary function types
    result = quote
        ReactiveMP.as_node_functional_form(::$dispatch_type)            = ReactiveMP.ValidNodeFunctionalForm()
        ReactiveMP.sdtype(::$dispatch_type)                             = (ReactiveMP.$node_type)()
    end
    # If there are any aliases, define the alias correction function
    if @capture(interface_aliases, aliases = aliases_)
        defined_aliases = map(alias_group -> Tuple(alias_group.args), aliases.args)
        all_aliases = vcat(all_aliases, defined_aliases)
    end

    check_all_symbol(all_aliases)

    first_interfaces = map(first, interface_alias_groups)

    for alias in all_aliases
        result = quote
            $result
            ReactiveMP.correct_interfaces(::$dispatch_type, nt::NamedTuple{$alias}) = NamedTuple{$(Tuple(first_interfaces))}(values(nt))
        end
    end
    return esc(result)
end

macro node(node_fform, node_type, node_interfaces)
    esc(quote 
        @node($node_fform, $node_type, $node_interfaces, nothing)
    end)
end