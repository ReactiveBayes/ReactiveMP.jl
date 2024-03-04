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
include("dependencies.jl")

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
    dependencies = collect_functional_dependencies(functionalform(factornode), getdependecies(options))
    return activate!(dependencies, factornode, options)
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

import .MacroHelpers

# Are still needed for the `@node` macro 
function make_node end
function interface_get_index end
function interface_get_name end

"""
    @node(fformtype, sdtype, interfaces_list)


`@node` macro creates a node for a `fformtype` type object. To obtain a list of available nodes use `?make_node`.

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

See `?make_node`.

See also: [`make_node`](@ref), [`Stochastic`](@ref), [`Deterministic`](@ref)
"""
macro node(fformtype, sdtype, interfaces_list)
    fbottomtype = MacroHelpers.bottom_type(fformtype)
    fuppertype  = MacroHelpers.upper_type(fformtype)

    @assert sdtype ∈ [:Stochastic, :Deterministic] "Invalid sdtype $(sdtype). Can be either Stochastic or Deterministic."

    @capture(interfaces_list, [interfaces_args__]) || error("Invalid interfaces specification.")

    interfaces = map(interfaces_args) do arg
        if @capture(arg, name_Symbol)
            return (name, [])
        elseif @capture(arg, (name_Symbol, aliases = [aliases__]))
            @assert all(a -> a isa Symbol && !isequal(a, name), aliases)
            return (name, aliases)
        else
            error("Interface specification should have a 'name' or (name, aliases = [ alias1, alias2,... ]) signature.")
        end
    end

    @assert length(interfaces) !== 0 "Node should have at least one interface."

    names   = map(d -> d[1], interfaces)
    aliases = map(d -> d[2], interfaces)

    foreach(names) do name
        @assert !occursin('_', string(name)) "Node interfaces names (and aliases) must not contain `_` symbol in them, found in $(name)."
    end

    foreach(Iterators.flatten(aliases)) do alias
        @assert !occursin('_', string(alias)) "Node interfaces names (and aliases) must not contain `_` symbol in them, found in $(alias)."
    end

    names_quoted_tuple     = Expr(:tuple, map(name -> Expr(:quote, name), names)...)
    names_indices          = Expr(:tuple, map(i -> i, 1:length(names))...)
    names_splitted_indices = Expr(:tuple, map(i -> Expr(:tuple, i), 1:length(names))...)
    names_indexed          = Expr(:tuple, map(name -> Expr(:call, :(ReactiveMP.indexed_name), name), names)...)

    interface_names       = map(name -> :(ReactiveMP.indexed_name($name)), names)
    interface_args        = map(name -> :($name), names)
    interface_connections = map(name -> :(ReactiveMP.connect!(node, $(Expr(:quote, name)), $name)), names)

    joined_interface_names = :(join((($(interface_names...)),), ", "))

    # Check that all arguments within interface refer to the unique var objects
    non_unique_error_sym = gensym(:non_unique_error_sym)
    non_unique_error_msg = :($non_unique_error_sym = (fformtype, names) -> """
                                                                           Non-unique variables used for the creation of the `$(fformtype)` node, which is disallowed.
                                                                           Check creation of the `$(fformtype)` with the `[ $(join(names, ", ")) ]` arguments.
                                                                           """)
    interface_uniqueness = map(enumerate(names)) do (index, name)
        names_without_current = skipindex(names, index)
        return quote
            if Base.in($(name), ($(names_without_current...),))
                Base.error($(non_unique_error_sym)($fformtype, $names_indexed))
            end
        end
    end

    # Here we create helpers function for GraphPPL.jl interfacing
    # They are used to convert interface names from `where { q = q(x, y)q(z) }` to an equivalent tuple respresentation, e.g. `((1, 2), (3, ))`
    # The general recipe to get a proper index is to call `interface_get_index(Val{ :NodeTypeName }, interface_get_name(Val{ :NodeTypeName }, Val{ :name_expression }))`
    interface_name_getters = map(enumerate(interfaces)) do (index, interface)
        name    = first(interface)
        aliases = last(interface)

        index_name_getter  = :(ReactiveMP.interface_get_index(::Type{Val{$(Expr(:quote, fbottomtype))}}, ::Type{Val{$(Expr(:quote, name))}}) = $(index))
        name_symbol_getter = :(ReactiveMP.interface_get_name(::Type{Val{$(Expr(:quote, fbottomtype))}}, ::Type{Val{$(Expr(:quote, name))}}) = $(Expr(:quote, name)))
        name_index_getter  = :(ReactiveMP.interface_get_name(::Type{Val{$(Expr(:quote, fbottomtype))}}, ::Type{Val{$index}}) = $(Expr(:quote, name)))

        alias_getters = map(aliases) do alias
            return :(ReactiveMP.interface_get_name(::Type{Val{$(Expr(:quote, fbottomtype))}}, ::Type{Val{$(Expr(:quote, alias))}}) = $(Expr(:quote, name)))
        end

        return quote
            $index_name_getter
            $name_symbol_getter
            $name_index_getter
            $(alias_getters...)
        end
    end

    # By default every argument passed to a factorisation option of the node is transformed by
    # `collect_factorisation` function to have a tuple like structure.
    # The default recipe is simple: for stochastic nodes we convert `FullFactorisation` and `MeanField` objects
    # to their tuple of indices equivalents. For deterministic nodes any factorisation is replaced by a FullFactorisation equivalent
    factorisation_collectors = if sdtype === :Stochastic
        quote
            ReactiveMP.collect_factorisation(::$fuppertype, ::Nothing)                      = ($names_indices,)
            ReactiveMP.collect_factorisation(::$fuppertype, factorisation::Tuple)           = factorisation
            ReactiveMP.collect_factorisation(::$fuppertype, ::ReactiveMP.FullFactorisation) = ($names_indices,)
            ReactiveMP.collect_factorisation(::$fuppertype, ::ReactiveMP.MeanField)         = $names_splitted_indices
        end

    elseif sdtype === :Deterministic
        quote
            ReactiveMP.collect_factorisation(::$fuppertype, ::Nothing)                      = ($names_indices,)
            ReactiveMP.collect_factorisation(::$fuppertype, factorisation::Tuple)           = ($names_indices,)
            ReactiveMP.collect_factorisation(::$fuppertype, ::ReactiveMP.FullFactorisation) = ($names_indices,)
            ReactiveMP.collect_factorisation(::$fuppertype, ::ReactiveMP.MeanField)         = ($names_indices,)
        end
    else
        error("Unreachable in @node macro.")
    end

    doctype   = rpad(fbottomtype, 30)
    docsdtype = rpad(sdtype, 15)
    docedges  = string(interfaces_list)
    doc       = """
        $doctype : $docsdtype : $docedges
    """

    res = quote
        ReactiveMP.as_node_functional_form(::$fuppertype)       = ReactiveMP.ValidNodeFunctionalForm()
        ReactiveMP.as_node_functional_form(::Type{$fuppertype}) = ReactiveMP.ValidNodeFunctionalForm()

        ReactiveMP.sdtype(::$fuppertype) = (ReactiveMP.$sdtype)()

        ReactiveMP.as_node_symbol(::$fuppertype) = $(QuoteNode(fbottomtype))

        @doc $doc function ReactiveMP.make_node(::Union{$fuppertype, Type{$fuppertype}}, options::FactorNodeCreationOptions)
            return ReactiveMP.FactorNode(
                $fbottomtype,
                $names_quoted_tuple,
                ReactiveMP.collect_factorisation($fbottomtype, ReactiveMP.factorisation(options)),
                ReactiveMP.collect_meta($fbottomtype, ReactiveMP.metadata(options)),
                ReactiveMP.collect_pipeline($fbottomtype, ReactiveMP.getpipeline(options))
            )
        end

        function ReactiveMP.make_node(::Union{$fuppertype, Type{$fuppertype}}, options::FactorNodeCreationOptions, $(interface_args...))
            node = ReactiveMP.make_node($fbottomtype, options)
            $(non_unique_error_msg)
            $(interface_uniqueness...)
            $(interface_connections...)
            return node
        end

        # Fallback method for unsupported number of arguments, e.g. if node expects 2 inputs, but only 1 was given
        function ReactiveMP.make_node(::Union{$fuppertype, Type{$fuppertype}}, options::FactorNodeCreationOptions, args...)
            ReactiveMP.make_node_incompatible_number_of_arguments_error($fuppertype, $fbottomtype, $interfaces, args)
        end

        $(interface_name_getters...)

        $factorisation_collectors
    end

    return esc(res)
end
