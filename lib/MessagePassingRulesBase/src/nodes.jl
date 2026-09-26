"""
    Stochastic()

The kind of a node with a density over its interfaces, `f(out | inputs)`, written
`type = Stochastic` in [`@define_factor_node`](@ref). Such a node has an average energy, and, when
it has no groups, a [`nodefunction`](@ref). Its clusters follow the graph's factorisation.
[`sdtype`](@ref)`(node)` returns `Stochastic()` for it.

See also [`Deterministic`](@ref).
"""
struct Stochastic end

"""
    Deterministic()

The kind of a node whose output is a function of its inputs, `out = f(inputs)`, written
`type = Deterministic` in [`@define_factor_node`](@ref). Its clusters are always its output and
the joint over its inputs, whatever the graph's factorisation, so it cannot require
`factorisation = :meanfield`. A rule reaches its function through [`getnodefn`](@ref).
[`sdtype`](@ref)`(node)` returns `Deterministic()` for it.

See also [`Stochastic`](@ref).
"""
struct Deterministic end

"""
    InterfaceSpec

One interface of a [`NodeSpec`](@ref), as [`@define_factor_node`](@ref) declares it. Fields:

- `name::Symbol`: the interface's name, `:μ`, or the group's, `:inputs` for `:inputs...`;
- `group::Bool`: whether it is a group of any number of members;
- `aliases::Tuple{Vararg{Symbol}}`: the other names a graph may use for it, `(:mean,)`.
"""
struct InterfaceSpec
    name::Symbol
    group::Bool
    aliases::Tuple{Vararg{Symbol}}
end

"""
    NodeSpec

A node as data, produced by [`@define_factor_node`](@ref) and returned by [`nodespec`](@ref). An
engine builds the node in a graph from it. The queries below read its fields; each keyword of the
macro is one field:

- `node`: the node itself, a type or a function;
- `type`: [`Stochastic`](@ref)`()` or [`Deterministic`](@ref)`()`, read by [`sdtype`](@ref);
- `interfaces`: the [`InterfaceSpec`](@ref)s in declaration order, read by [`interfaces`](@ref),
  [`interface_groups`](@ref) and [`alias_interface`](@ref);
- `algorithm`: the algorithm value rules run under by default, read by
  [`default_algorithm`](@ref);
- `static_inputs`: `:none` or `:fold`, read by [`static_inputs`](@ref);
- `matched_groups`, `min_group_length`, `factorisation`: what the node requires of a graph, read
  by [`matched_groups`](@ref), [`min_group_length`](@ref) and [`required_factorisation`](@ref);
- `initial_messages`: `name => message` pairs, read by [`initial_messages`](@ref);
- `file`, `line`: where it was declared.

It shows itself as a summary at the REPL and as a table in a notebook.
"""
struct NodeSpec
    node::Any
    type::Union{Stochastic, Deterministic}
    interfaces::Tuple{Vararg{InterfaceSpec}}
    algorithm::AbstractAlgorithm
    static_inputs::Symbol
    matched_groups::Tuple{Vararg{Tuple{Vararg{Symbol}}}}
    min_group_length::Int
    factorisation::Symbol
    initial_messages::Tuple{Vararg{Pair{Symbol}}}
    file::Symbol
    line::Int
end

"""
    nodespec(node) -> NodeSpec

The [`NodeSpec`](@ref) [`@define_factor_node`](@ref) declared for `node`, a type or a function.
Every other node query reads it.

# Throws
`MethodError` for a node that was never declared.
"""
function nodespec end

"""
    interfaces(node) -> Tuple{Vararg{Symbol}}

The interface names of `node`, in declaration order; a group appears once, by its name.
Throws a `MethodError` for an undeclared node, as every node query does.

```jldoctest
julia> struct Mix end

julia> @define_factor_node(node = Mix, type = Stochastic, interfaces = [:out, :switch, :m...])

julia> MessagePassingRulesBase.interfaces(Mix), MessagePassingRulesBase.interface_groups(Mix)
((:out, :switch, :m), (:m,))
```
"""
interfaces(node) = map(i -> i.name, nodespec(node).interfaces)

"""
    interface_groups(node) -> Tuple{Vararg{Symbol}}

The names of the groups of `node`, the interfaces declared as `:m...`, in declaration order;
empty for a node without groups.
"""
interface_groups(node) = map(i -> i.name, filter(i -> i.group, nodespec(node).interfaces))

"""
    sdtype(node) -> Union{Stochastic, Deterministic}

The kind of `node`: [`Stochastic`](@ref)`()` or [`Deterministic`](@ref)`()`, as declared.
"""
sdtype(node) = nodespec(node).type

"""
    default_algorithm(node) -> AbstractAlgorithm

The algorithm value rules for `node` run under unless a call or a graph gives another: the
node's declared `algorithm`, instantiated, and [`DefaultAlgorithm`](@ref)`()` for a node that
declares none. A rule that omits `algorithm` is defined for its type.
"""
default_algorithm(node) = nodespec(node).algorithm

"""
    static_inputs(node) -> Symbol

How the node treats inputs connected to constants and data. `:none` treats them like any
other input. `:fold` folds them into the node function, reached as
[`getnodefn`](@ref)`(ctx.node, target)`, and every update waits until they are available. Which
inputs are static is known only from the graph, so the engine does the folding and the waiting.
"""
static_inputs(node) = nodespec(node).static_inputs

"""
    matched_groups(node) -> Tuple

The sets of groups that must have as many members as each other, as tuples of group names:
`((:m, :p),)` for a mixture whose means and precisions come in pairs. Empty when the node
declares none.
"""
matched_groups(node) = nodespec(node).matched_groups

"""
    min_group_length(node) -> Int

The fewest members any group of `node` may have: 1 unless declared, 2 for a mixture, 0 for
a node whose group may be empty, as `DiscreteTransition`'s `T`.
"""
min_group_length(node) = nodespec(node).min_group_length

"""
    required_factorisation(node) -> Symbol

The factorisations `node` accepts. `:any` accepts every one; `:meanfield` only clusters of one
interface each, for a node whose rules are variational whatever the factorisation, such as
a mixture. The engine checks it when it creates the node.
"""
required_factorisation(node) = nodespec(node).factorisation

"""
    initial_messages(node) -> Tuple

The messages `node` seeds its interfaces with, as `name => message` pairs: for a rule that reads
the message on its own edge, the value it starts from. The engine sets each on the node's
inbound message of that interface at activation, unless one was set there already, so a
user's initialisation wins. Empty unless declared.
"""
initial_messages(node) = nodespec(node).initial_messages

"""
    alias_interface(node, name::Symbol) -> Symbol

The declared interface `name` refers to: `name` itself for an interface, or the interface it is
an alias of. An engine calls it on the names a graph uses.

# Throws
`ArgumentError` naming the node's interfaces when `name` is neither an interface nor an alias.
"""
function alias_interface(node, name::Symbol)
    for interface in nodespec(node).interfaces
        (interface.name === name || name in interface.aliases) && return interface.name
    end
    throw(ArgumentError("$node has no interface or alias `$name`; its interfaces are $(interfaces(node))"))
end

"""
    nodefunction(node) -> Function

The log-density of a stochastic node without groups, as a function of keyword arguments named
after its interfaces: `(; out, μ, v) -> logpdf(node(μ, v), out)`. [`@define_factor_node`](@ref)
defines it for such a node, which must be callable as a distribution of its other interfaces.
The rule fallback [`NodeFunctionRuleFallback`](@ref) and rule verification use it.

# Throws
`MethodError` for a deterministic node, a node with groups, or an undeclared one.

```julia
f = MessagePassingRulesBase.nodefunction(NormalMeanVariance)
f(out = 1.0, μ = 0.0, v = 2.0)   # logpdf(NormalMeanVariance(0.0, 2.0), 1.0)
```
"""
function nodefunction end

"""
    getnodefn(node, target)

The function a deterministic node computes, as a rule body needs it: for `Target(:out)` the
forward function of the free inputs, with any static inputs already folded in. A rule
reaches it as `getnodefn(ctx.node, Target(:out))` after declaring `ctx = (:node,)`.

The base package declares it and defines no methods: `node` is the engine's own node
object, which owns the function and its static inputs, so the engine implements it. A known
inverse is not the node's but its algorithm's, which a rule reads from `algo`.
"""
function getnodefn end
