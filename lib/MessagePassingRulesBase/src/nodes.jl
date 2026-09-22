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
    dependencies::Any
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
    groups(node)

The names of the variadic interface groups of `node`.
"""
groups(node) = map(i -> i.name, filter(i -> i.group, nodespec(node).interfaces))

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
