export Deterministic, Stochastic, isdeterministic, isstochastic, sdtype
export MeanField, FullFactorisation, Marginalisation, MomentMatching
export functionalform, interfaces, factorisation, localmarginals, localmarginalnames, metadata
export FactorNode, factornode
export @node

using Rocket
using TupleTools
using MacroTools

import Rocket: getscheduler

import Base: show, +, push!, iterate, IteratorSize, IteratorEltype, eltype, length, size
import Base: getindex, setindex!, firstindex, lastindex

## 

function make_node end # TODO (bvdmitri): remove this

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
include("dependencies.jl")

abstract type AbstractFactorNode end

## Generic Factor node new code start

"""
    GenericFactorNode(functionalform, interfaces)

Generic factor node object that represents a factor node with a given `functionalform` and `interfaces`.
"""
struct FactorNode{F, I} <: AbstractFactorNode
    interfaces::I

    function FactorNode(::Type{F}, interfaces::I) where {F, I <: Tuple}
        return new{F, I}(interfaces)
    end
end

factornode(::Type{F}, interfaces::I) where {F, I} = FactorNode(F, __prepare_interfaces_generic(interfaces))
factornode(::F, interfaces::I) where {F <: Function, I} = FactorNode(F, __prepare_interfaces_generic(interfaces))

functionalform(factornode::FactorNode{F}) where {F} = F
getinterfaces(factornode::FactorNode) = factornode.interfaces
getinterface(factornode::FactorNode, index) = factornode.interfaces[index]

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

function activate!(factornode::FactorNode, options::FactorNodeActivationOptions)
    dependencies = collect_functional_dependencies(functionalform(factornode), getdependecies(options))
    return activate!(dependencies, factornode, options)
end

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

function generate_node_expression(node_fform, node_type, node_interfaces, interface_aliases)
    # Assert that the node type is either Stochastic or Deterministic, and that all interfaces are symbols
    @assert node_type ∈ [:Stochastic, :Deterministic]
    @assert length(node_interfaces.args) > 0

    interface_alias_groups = map(alias_group, node_interfaces.args)
    all_aliases = vec(collect(Iterators.product(interface_alias_groups...)))

    # Determine whether we should dispatch on `typeof($fform)` or `Type{$node_fform}`
    dispatch_type = if @capture(node_fform, typeof(fform_))
        :(typeof($fform))
    else
        :(Type{$node_fform})
    end

    # If there are any aliases, define the alias correction function
    if @capture(interface_aliases, aliases = aliases_)
        defined_aliases = map(alias_group -> Tuple(alias_group.args), aliases.args)
        all_aliases = vcat(all_aliases, defined_aliases)
    end

    check_all_symbol(all_aliases)

    first_interfaces = map(first, interface_alias_groups)

    alias_corrections = Expr(:block)
    alias_corrections.args = map(all_aliases) do alias
        :(ReactiveMP.correct_interfaces(::$dispatch_type, nt::NamedTuple{$alias}) = NamedTuple{$(Tuple(first_interfaces))}(values(nt)))
    end

    # Define the necessary function types
    result = quote
        ReactiveMP.as_node_functional_form(::$dispatch_type)                     = ReactiveMP.ValidNodeFunctionalForm()
        ReactiveMP.sdtype(::$dispatch_type)                                      = (ReactiveMP.$node_type)()
        ReactiveMP.collect_factorisation(::$dispatch_type, factorisation::Tuple) = factorisation

        $alias_corrections
    end

    return result
end

macro node(node_fform, node_type, node_interfaces, interface_aliases)
    return esc(generate_node_expression(node_fform, node_type, node_interfaces, interface_aliases))
end

macro node(node_fform, node_type, node_interfaces)
    return esc(generate_node_expression(node_fform, node_type, node_interfaces, nothing))
end