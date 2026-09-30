```@meta
CurrentModule = MessagePassingRulesBase
```

# Defining nodes

You declare a [factor node](@ref glossary-factor-node) once, with [`@define_factor_node`](@ref),
before any of its rules. The declaration names the node and says whether it is stochastic or
deterministic. It lists the node's [interfaces](@ref glossary-interface), its named edges. It
may also give the node's own [algorithm](@ref glossary-algorithm), its
[dependencies](@ref glossary-dependencies), its [initial messages](@ref glossary-initial-message)
and what the node requires of a graph.

```@example nodes
using MessagePassingRulesBase

struct Mixture end

@define_factor_node(
    node = Mixture,
    type = Stochastic,
    interfaces = [:out, (:switch, aliases = [:s]), :inputs...],
    min_group_length = 2,
)

MessagePassingRulesBase.nodespec(Mixture)
```

The node itself is a type, as `Mixture` here or `NormalMeanVariance` in a rule package, or a
function, as `+`. The same value names the node in every rule and in a graph. The declaration
draws itself: this node has an output, an interface `switch` with the alias `s`, and a group of
inputs with at least two members. [Your first node](@ref tutorial-first-node) declares a node
and writes its rules step by step. The [Keyword reference](@ref keyword-reference) lists every
keyword of the macro.

## Interfaces

You list the interfaces in order, the output first by convention. An interface may have
aliases, other names a graph may use for it: `(:μ, aliases = [:mean])`.

A trailing `...` declares a [group](@ref glossary-group). A group has any number of members,
`(:inputs, 1)`, `(:inputs, 2)` and so on, and a graph gives them as a whole. A mixture's
components and a sum's summands are groups. A rule targets any member with
`target = (:inputs, k)`, and it reads a group as a tuple in member order.
[A deterministic node with a group](@ref tutorial-groups) writes the rules of such a node.

Names may contain underscores, since nothing in the package joins names together.

## Kinds

A [`Stochastic`](@ref) node has a density over its interfaces, `f(out | inputs)`. Its
[clusters](@ref glossary-cluster) follow the graph's [factorisation](@ref glossary-factorisation),
and it has an [average energy](@ref glossary-average-energy). When a stochastic node has no
groups, the macro also defines its log-density as [`nodefunction`](@ref). Rule verification and
the [`NodeFunctionRuleFallback`](@ref) use it.

A [`Deterministic`](@ref) node computes its output from its inputs, `out = f(inputs)`. Its
clusters are always its output and the joint over its inputs, whatever the factorisation. A
rule reaches the function through [`getnodefn`](@ref)`(ctx.node, Target(:out))`. The engine
implements that call, because the engine's node owns the function and any static inputs folded
into it.

```@docs
@define_factor_node
Stochastic
Deterministic
getnodefn
```

## What a node requires of a graph

A node may state what it requires of the graph it is placed in. An engine checks these
requirements when it creates the node. A malformed graph is then an error at creation, rather
than a rule that silently reads fewer components.

- `matched_groups = [(:m, :p)]`: the named groups have as many members as each other.
- `min_group_length = 2`: every group has at least two members. `0` allows an empty group.
- `factorisation = :meanfield`: the graph gives every interface a cluster of its own. A node
  whose rules are variational whatever the factorisation declares it.

`static_inputs = :fold` asks the engine to fold the inputs that are connected to constants and
data into the node's function, which a rule reads through [`getnodefn`](@ref). An engine builds
such a node with its function.

## The declaration as data

The macro produces a [`NodeSpec`](@ref), which [`nodespec`](@ref) returns. The card at the top of
this page is that `NodeSpec`. Every query below reads it:

```@repl nodes
MessagePassingRulesBase.interfaces(Mixture)
MessagePassingRulesBase.interface_groups(Mixture)
MessagePassingRulesBase.alias_interface(Mixture, :s)
```

```@docs
MessagePassingRulesBase.NodeSpec
MessagePassingRulesBase.InterfaceSpec
MessagePassingRulesBase.nodespec
MessagePassingRulesBase.interfaces
MessagePassingRulesBase.interface_groups
MessagePassingRulesBase.alias_interface
MessagePassingRulesBase.sdtype
MessagePassingRulesBase.static_inputs
MessagePassingRulesBase.matched_groups
MessagePassingRulesBase.min_group_length
MessagePassingRulesBase.required_factorisation
MessagePassingRulesBase.nodefunction
```

The node's algorithm, its dependencies and its initial messages are on
[Algorithms and dependencies](@ref).
