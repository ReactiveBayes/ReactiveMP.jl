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

"""
    GenericFactorNode(functionalform, interfaces)

Generic factor node object that represents a factor node with a given `functionalform` and `interfaces`.
"""
struct FactorNode{F, I, C} <: AbstractFactorNode
    interfaces::I
    localclusters::C

    function FactorNode(::Type{F}, interfaces::I, factorization) where {F, I}
        clusters = FactorNodeLocalClusters(interfaces, factorization)
        C = typeof(clusters)
        return new{F, I, C}(interfaces, clusters)
    end
end

factornode(::Type{F}, interfaces::I, factorization) where {F, I} = FactorNode(F, __prepare_interfaces_generic(F, interfaces), factorization)
factornode(::F, interfaces::I, facatorization) where {F <: Function, I} = FactorNode(F, __prepare_interfaces_generic(F, interfaces), factorization)

functionalform(factornode::FactorNode{F}) where {F} = F
getinterfaces(factornode::FactorNode) = factornode.interfaces
getinterface(factornode::FactorNode, index) = factornode.interfaces[index]
getlocalclusters(factornode::FactorNode) = factornode.localclusters
sdtype(factornode::FactorNode) = sdtype(functionalform(factornode))

# Takes a named tuple of abstract variables and converts to a tuple of NodeInterfaces with the same order
function __prepare_interfaces_generic(::Type{F}, interfaces::AbstractVector) where {F}
    return map(enumerate(interfaces)) do (index, (name, variable))
        return NodeInterface(alias_interface(F, index, name), variable)
    end
end

## activate!

struct FactorNodeActivationOptions{M, D, P, A, S}
    metadata::M
    dependencies::D
    pipeline::P
    addons::A
    scheduler::S
end

getmetadata(options::FactorNodeActivationOptions) = options.metadata
getdependecies(options::FactorNodeActivationOptions) = options.dependencies
getpipeline(options::FactorNodeActivationOptions) = options.pipeline
getaddons(options::FactorNodeActivationOptions) = options.addons
getscheduler(options::FactorNodeActivationOptions) = options.scheduler

function activate!(factornode::FactorNode, options::FactorNodeActivationOptions)
    initialize_clusters!(getlocalclusters(factornode), factornode, options)
    dependencies = collect_functional_dependencies(functionalform(factornode), getdependecies(options))
    return activate!(dependencies, factornode, options)
end

import .MacroHelpers

function alias_interface end

extract_interface(s::Symbol) = (s, [])

function extract_interface(e::Expr)
    if @capture(e, (s_, aliases = [aliases__]))
        if !all(alias -> alias isa Symbol, aliases)
            error(lazy"Aliases should be pure symbols. Got expression in $(aliases).")
        end
        return (s, aliases)
    else
        error(lazy"Unknown interface specification: $(e)")
    end
end

function generate_node_expression(node_fform, node_type, node_interfaces)
    # Assert that the node type is either Stochastic or Deterministic, and that all interfaces are symbols
    @assert node_type ∈ [:Stochastic, :Deterministic]
    @assert length(node_interfaces.args) > 0

    interfaces = map(extract_interface, node_interfaces.args)

    # Determine whether we should dispatch on `typeof($fform)` or `Type{$node_fform}`
    dispatch_type = if @capture(node_fform, typeof(fform_))
        :(typeof($fform))
    else
        :(Type{$node_fform})
    end

    alias_corrections = Expr(:block)
    alias_corrections.args = map(enumerate(interfaces)) do (index, (name, aliases))
        # The `index` and `name` variables are defined further in the `alias_interface` function
        quote
            if index === $index && (name === $(QuoteNode(name)) || Base.in(name, ($(map(QuoteNode, aliases)...),)))
                return $(QuoteNode(name))
            end
        end
    end

    # Define the necessary function types
    result = quote
        ReactiveMP.as_node_functional_form(::$dispatch_type)                     = ReactiveMP.ValidNodeFunctionalForm()
        ReactiveMP.sdtype(::$dispatch_type)                                      = (ReactiveMP.$node_type)()
        ReactiveMP.collect_factorisation(::$dispatch_type, factorisation::Tuple) = factorisation

        function ReactiveMP.alias_interface(dispatch_type::$dispatch_type, index, name)
            $alias_corrections
            # If we do not return from the `alias_corrections` we throw an error
            error(lazy"Don't know how to alias interface $(name) in $(index) for $(dispatch_type)")
        end
    end

    return result
end

macro node(node_fform, node_type, node_interfaces)
    return esc(generate_node_expression(node_fform, node_type, node_interfaces))
end