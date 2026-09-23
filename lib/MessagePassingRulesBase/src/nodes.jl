"""
    Stochastic()

A node with a density over its interfaces.
"""
struct Stochastic end

"""
    Deterministic()

A node whose output is a function of its inputs.
"""
struct Deterministic end

"""
    InterfaceSpec

One declared interface: its name, whether it is a variadic group, and its aliases.
"""
struct InterfaceSpec
    name::Symbol
    group::Bool
    aliases::Tuple{Vararg{Symbol}}
end

"""
    NodeSpec

A node as data, produced by [`@define_factor_node`](@ref).
"""
struct NodeSpec
    node::Any
    type::Union{Stochastic, Deterministic}
    interfaces::Tuple{Vararg{InterfaceSpec}}
    algorithm::AbstractAlgorithm
    static_inputs::Symbol
    file::Symbol
    line::Int
end

"""
    nodespec(node)

The [`NodeSpec`](@ref) declared for `node`.
"""
function nodespec end

"""
    interfaces(node)

The interface names of `node`, in declaration order; a group appears once, by name.
"""
interfaces(node) = map(i -> i.name, nodespec(node).interfaces)

"""
    interface_groups(node)

The names of the variadic interface groups of `node`.
"""
interface_groups(node) = map(i -> i.name, filter(i -> i.group, nodespec(node).interfaces))

"""
    sdtype(node)

`Stochastic()` or `Deterministic()`.
"""
sdtype(node) = nodespec(node).type

"""
    default_algorithm(node)

The algorithm rules for `node` run under unless one is given.
"""
default_algorithm(node) = nodespec(node).algorithm

"""
    static_inputs(node)

How the node treats inputs connected to constants and data. `:none` treats them like any
other input. `:fold` folds them into the node function, reached as
[`getnodefn`](@ref)`(ctx.node, target)`, and every update waits until they are available. Which inputs are static is known only
from the graph, so the engine does the folding and the waiting.
"""
static_inputs(node) = nodespec(node).static_inputs

"""
    alias_interface(node, name)

The declared interface `name` refers to, directly or as an alias.
"""
function alias_interface(node, name::Symbol)
    for interface in nodespec(node).interfaces
        (interface.name === name || name in interface.aliases) && return interface.name
    end
    throw(ArgumentError("$node has no interface or alias `$name`; its interfaces are $(interfaces(node))"))
end

"""
    nodefunction(node)

The log-density of a stochastic node without groups, as a function of keyword arguments
named after its interfaces.
"""
function nodefunction end

"""
    getnodefn(node, target)

The function a deterministic node computes, as a rule body needs it: for `Target(:out)` the
forward function of the free inputs, with any static inputs already folded in. A rule
reaches it as `getnodefn(ctx.node, Target(:out))` after declaring `ctx = (:node,)`.

The base package declares it and defines no methods: `node` is the engine's own node
object, which owns the function and its static inputs, so the engine implements it. It
replaces v6's `nodefunction(node, meta, Val(:out))`. A known inverse, v6's
`(Val(:in), k)`, is not the node's but its algorithm's, which a rule reads from `algo`.
"""
function getnodefn end
