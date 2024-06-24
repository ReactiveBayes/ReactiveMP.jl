export Deterministic, Stochastic, isdeterministic, isstochastic, sdtype
export Marginalisation, MomentMatching
export functionalform, getinterfaces, factorisation, localmarginals, localmarginalnames
export FactorNode, factornode
export @node

using Rocket
using TupleTools
using MacroTools

import Rocket: getscheduler

import Base: show, +, push!, iterate, IteratorSize, IteratorEltype, eltype, length, size
import Base: getindex, setindex!, firstindex, lastindex

## Node traits

"""
    PredefinedNodeFunctionalForm

Trait specification for an object that has been marked as a valid factor node with the `@node` macro.

See also: [`ReactiveMP.is_predefined_node`](@ref), [`ReactiveMP.UndefinedNodeFunctionalForm`](@ref)
"""
struct PredefinedNodeFunctionalForm end

"""
    UndefinedNodeFunctionalForm

Trait specification for an object that has **not** been marked as a factor node with the `@node` macro.
Note that it does not necessarily mean that the object is not a valid factor node, but rather that it has not been marked as such.
The ReactiveMP inference engine support arbitrary deterministic function as factor nodes, but they require manual specification of the approximation method.

See also: [`ReactiveMP.is_predefined_node`](@ref), [`ReactiveMP.PredefinedNodeFunctionalForm`](@ref)
"""
struct UndefinedNodeFunctionalForm end

"""
    is_predefined_node(object)

Determines if the `object` has been marked as a factor node with the `@node` macro.
Returns either `PredefinedNodeFunctionalForm()` or `UndefinedNodeFunctionalForm()`.

See also: [`ReactiveMP.PredefinedNodeFunctionalForm`](@ref), [`ReactiveMP.UndefinedNodeFunctionalForm`](@ref)
"""
function is_predefined_node end

# By default all objects are not marked as nodes
is_predefined_node(some) = UndefinedNodeFunctionalForm()

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
    nodefunction(::Type{T}) where {T}

Returns a function that represents a node of type `T`. 
The function typically takes arguments that represent the node's input and output variables in the same order as defined in the `@node` macro.
"""
function nodefunction end

"""
    sdtype(object)

Returns either `Deterministic` or `Stochastic` for a given object (if defined).

See also: [`Deterministic`](@ref), [`Stochastic`](@ref), [`isdeterministic`](@ref), [`isstochastic`](@ref)
"""
sdtype(any) = error("Unknown if an object of type `$(typeof(any))` is stochastic or deterministic.")

# Any `Type` is considered to be a deterministic mapping unless stated otherwise (By convention, any `Distribution` type is not deterministic)
# E.g. `Matrix` is not an instance of the `Function` abstract type, however we would like to pretend it is a deterministic function
sdtype(::Type{T}) where {T}    = Deterministic()
sdtype(::Function)             = Deterministic()
sdtype(::Type{<:Distribution}) = Stochastic()
sdtype(::Distribution)         = Stochastic()

"""
    as_node_symbol(type)

Returns a symbol associated with a node `type`.
"""
function as_node_symbol end

as_node_symbol(fn::F) where {F <: Function} = Symbol(fn)

"""
    collect_factorisation(nodetype, factorisation)

This function converts given factorisation to a correct internal factorisation representation for a given node.
"""
function collect_factorisation end

"""
    collect_meta(nodetype, meta)

This function converts given meta object to a correct internal meta representation for a given node.
Fallbacks to `default_meta` in case if meta is `nothing`.

See also: [`default_meta`](@ref), [`FactorNode`](@ref)
"""
function collect_meta end

collect_meta(fform::F, ::Nothing) where {F} = default_meta(fform)
collect_meta(fform::F, meta::Any) where {F} = meta

"""
    default_meta(nodetype)

Returns default meta object for a given node type.

See also: [`collect_meta`](@ref), [`FactorNode`](@ref)
"""
default_meta(fform) = nothing

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
    fform::F
    interfaces::I
    localclusters::C

    FactorNode(fform::Type{F}, interfaces::I, localclusters::C) where {F, I, C} = new{Type{F}, I, C}(fform, interfaces, localclusters)
    FactorNode(fform::F, interfaces::I, localclusters::C) where {F <: Function, I, C} = new{F, I, C}(fform, interfaces, localclusters)
end

function factornode(fform::F, interfaces::I, factorization) where {F, I}
    return factornode(is_predefined_node(fform), fform, interfaces, factorization)
end

# `PredefinedNodeFunctionalForm` are generally the nodes that are defined with the `@node` macro
# The `UndefinedNodeFunctionalForm` nodes can be created as well, but only if the `fform` is a `Function` (see `predefined/delta.jl`)
function factornode(::PredefinedNodeFunctionalForm, fform::F, interfaces::I, factorization) where {F, I}
    processed_interfaces = prepare_interfaces_generic(fform, interfaces)
    localclusters = FactorNodeLocalClusters(processed_interfaces, collect_factorisation(fform, factorization))
    return FactorNode(fform, processed_interfaces, localclusters)
end

functionalform(factornode::FactorNode) = factornode.fform
getinterfaces(factornode::FactorNode) = factornode.interfaces
getinterface(factornode::FactorNode, index) = factornode.interfaces[index]
# `getinboundinterfaces` skips the first interface, which is assumed to be the output interface
getinboundinterfaces(factornode::FactorNode) = view(factornode.interfaces, (firstindex(factornode.interfaces) + 1):lastindex(factornode.interfaces))
getlocalclusters(factornode::FactorNode) = factornode.localclusters
sdtype(factornode::FactorNode) = sdtype(functionalform(factornode))

interfaceindex(factornode::FactorNode, iname::Symbol)                         = findfirst(interface -> name(interface) === iname, getinterfaces(factornode))
interfaceindices(factornode::FactorNode, iname::Symbol)                       = (interfaceindex(factornode, iname),)
interfaceindices(factornode::FactorNode, inames::NTuple{N, Symbol}) where {N} = map(iname -> interfaceindex(factornode, iname), inames)

function prepare_interfaces_generic(fform::F, interfaces::AbstractVector) where {F}
    prepare_interfaces_check_nonempty(fform, interfaces)
    prepare_interfaces_check_adjacent_duplicates(fform, interfaces)
    prepare_interfaces_check_numarguments(fform, interfaces)
    return map(enumerate(interfaces)) do (index, (name, variable))
        return NodeInterface(alias_interface(fform, index, name), variable)
    end
end

function prepare_interfaces_check_nonempty(fform, interfaces)
    length(interfaces) > 0 || error(lazy"At least one argument is required for a factor node. Got none for `$(fform)`")
end

function prepare_interfaces_check_adjacent_duplicates(fform, interfaces)
    # Here we create an iterator that checks ONLY adjacent interfaces 
    # The reason here is that we don't want to check all possible combinations of all input interfaces
    # because that would require allocating an intermediate storage for `Set`, which would harm the 
    # performance of nodes creation. The `zip(interfaces, Iterators.drop(interfaces, 1))` creates a generic 
    # iterator of adjacent interface pairs
    foreach(zip(interfaces, Iterators.drop(interfaces, 1))) do (left, right)
        lname, _ = left
        rname, _ = right
        if isequal(lname, rname)
            error(
                lazy"`$fform` has duplicate entry for interface `$lname`. Did you pass an array (e.g. `x`) instead of an array element (e.g. `x[i]`)? Check your variable indices."
            )
        end
    end
end

function prepare_interfaces_check_numarguments(fform::F, interfaces) where {F}
    prepare_interfaces_check_num_inputarguments(fform, inputinterfaces(fform), interfaces)
end

function prepare_interfaces_check_num_inputarguments(fform, inputinterfaces::Val{Input}, interfaces) where {Input}
    (length(interfaces) - 1) === length(Input) ||
        error(lazy"Expected $(length(Input)) input arguments for `$(fform)`, got $(length(interfaces) - 1): $(join(map(first, Iterators.drop(interfaces, 1)), \", \"))")
end

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
    dependencies = collect_functional_dependencies(functionalform(factornode), getdependecies(options))
    initialize_clusters!(getlocalclusters(factornode), dependencies, factornode, options)
    return activate!(dependencies, factornode, options)
end

import .MacroHelpers

"""
    interfaces(fform)

Returns a `Val` object with a tuple of interface names for a given factor node type. Returns `nothing` for unknown functional form.
"""
interfaces(fform) = nothing

"""
    inputinterfaces(fform)

Similar to `interfaces`, but returns a `Val` object with a tuple of **input** interface names for a given factor node type. Returns `nothing` for unknown functional form.
"""
inputinterfaces(fform) = nothing

"""
    alias_interface(factor_type, index, name)

Converts the given `name` to a correct interface name for a given factor node type and index.
"""
function alias_interface end

node_expression_extract_interface(s::Symbol) = (s, [])

function node_expression_extract_interface(e::Expr)
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

    interfaces = map(node_expression_extract_interface, node_interfaces.args)

    # Determine whether we should dispatch on `typeof($fform)` or `Type{$node_fform}`
    dispatch_type = if @capture(node_fform, typeof(fform_))
        :(typeof($fform))
    else
        :(Type{<:$node_fform})
    end

    foreach(interfaces) do (name, aliases)
        @assert !occursin('_', string(name)) "Node interfaces names (and aliases) must not contain `_` symbol in them, found in `$(name)`."
        foreach(aliases) do alias
            @assert !occursin('_', string(alias)) "Node interfaces names (and aliases) must not contain `_` symbol in them, found in `$(alias)`."
        end
    end

    alias_corrections = Expr(:block)
    alias_corrections.args = map(enumerate(interfaces)) do (index, (name, aliases))
        # The `index` and `name` variables are defined further in the `alias_interface` function
        quote
            # TODO: (bvdmitri) maybe reserving `in` here is not a good idea, discuss with Wouter
            if index === $index && (name === :in || name === $(QuoteNode(name)) || Base.in(name, ($(map(QuoteNode, aliases)...),)))
                return $(QuoteNode(name))
            end
        end
    end

    collect_factorisation_fn = if node_type == :Stochastic
        :(ReactiveMP.collect_factorisation(::$dispatch_type, factorisation::Tuple) = factorisation)
    else
        :(ReactiveMP.collect_factorisation(::$dispatch_type, factorisation::Tuple) = ($(ntuple(identity, length(interfaces))),))
    end

    doctype   = rpad(dispatch_type, 30)
    docsdtype = rpad(node_type, 15)
    docedges  = join(map(((name, aliases),) -> string(name, !isempty(aliases) ? string(" (or ", join(aliases, ", "), ")") : ""), interfaces), ", ")
    doc       = """    
        $doctype : $docsdtype : $docedges
    The `$(node_fform)` has been marked as a valid `$(node_type)` factor node with the `@node` macro with `[ $(docedges) ]` interfaces.
    """

    # For `Stochastic` nodes the `nodefunctions` are pre-generated automatically 
    #   by calling the `corresponding` logpdf
    nodefunctions = if node_type == :Stochastic
        nodefunctionargnames = first.(interfaces)

        # The very first function is a generic method that only accepts type and returns 
        # a function that fallbacks to calculate the logpdf of the distribution
        fncollection = [
            :(
                ReactiveMP.nodefunction(::$dispatch_type) =
                    (; $(nodefunctionargnames...)) -> ReactiveMP.BayesBase.logpdf(($node_fform)($(nodefunctionargnames[2:end]...)), $(nodefunctionargnames[1]))
            )
        ]

        # The rest are individual node functions in each direction
        for interface in interfaces
            interfacename = first(interface)
            edgespecificfn = :(
                ReactiveMP.nodefunction(::$dispatch_type, ::Val{$(QuoteNode(interfacename))}; kwargs...) = begin
                    return let ckwargs = kwargs
                        ($interfacename) -> ReactiveMP.nodefunction($node_fform)(; $interfacename = $interfacename, ckwargs...)
                    end
                end
            )
            push!(fncollection, edgespecificfn)
        end

        _block = Expr(:block)
        _block.args = fncollection
        _block
    else
        :(nothing)
    end

    # Define the necessary function types
    result = quote
        @doc $doc ReactiveMP.is_predefined_node(::$dispatch_type) = ReactiveMP.PredefinedNodeFunctionalForm()

        ReactiveMP.sdtype(::$dispatch_type)          = (ReactiveMP.$node_type)()
        ReactiveMP.interfaces(::$dispatch_type)      = Val($(Tuple(map(first, interfaces))))
        ReactiveMP.inputinterfaces(::$dispatch_type) = Val($(Tuple(map(first, skipindex(interfaces, 1)))))

        $collect_factorisation_fn
        $nodefunctions

        function ReactiveMP.alias_interface(dispatch_type::$dispatch_type, index, name)
            $alias_corrections
            # If we do not return from the `alias_corrections` we throw an error
            error(lazy"Don't know how to alias interface $(name) in $(index) for $(dispatch_type)")
        end
    end

    return result
end

"""
    @node(fformtype, sdtype, interfaces_list)


`@node` macro creates a node for a `fformtype` type object. To obtain a list of predefined nodes use `?is_predefined_node`.

# Arguments

- `fformtype`: Either an existing type identifier, e.g. `Normal` or a function type identifier, e.g. `typeof(+)`
- `sdtype`: Either `Stochastic` or `Deterministic`. Defines the type of the functional relationship
- `interfaces_list`: Defines a fixed list of edges of a factor node, by convention the first element should be `out`. Example: `[ out, mean, variance ]`

Note: `interfaces_list` must not include names that contain `_` symbol in them, as it is reserved to identify joint posteriors around the node object.

# Examples
```julia

struct MyNormalDistribution
    mean :: Float64
    var  :: Float64
end

@node MyNormalDistribution Stochastic [ out, mean, var ]
```

```julia

@node typeof(+) Deterministic [ out, in1, in2 ]
```

# List of available nodes

See `?is_predefined_node`.
"""
macro node(node_fform, node_type, node_interfaces)
    return esc(generate_node_expression(node_fform, node_type, node_interfaces))
end
