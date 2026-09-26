```@meta
CurrentModule = MessagePassingRulesBase
```

# Defining nodes

A factor node is declared once, with [`@define_factor_node`](@ref), before any of its rules. The
declaration names the node, says whether it is stochastic or deterministic, lists its
interfaces, and optionally gives its own algorithm, its dependencies, its initial messages and
what it requires of a graph. The node itself is a type, `node = NormalMeanVariance`, or a
function, `node = +`; the same value names it in every rule and in a graph.

```jldoctest nodes
julia> using MessagePassingRulesBase

julia> struct Mixture end

julia> @define_factor_node(
           node = Mixture,
           type = Stochastic,
           interfaces = [:out, (:switch, aliases = [:s]), :inputs...],
           min_group_length = 2,
       )

julia> MessagePassingRulesBase.interfaces(Mixture)
(:out, :switch, :inputs)

julia> MessagePassingRulesBase.interface_groups(Mixture)
(:inputs,)

julia> MessagePassingRulesBase.alias_interface(Mixture, :s)
:switch
```

## Interfaces

The interfaces are listed in order, the output first by convention. An interface may have
aliases, `(:μ, aliases = [:mean])`, other names a graph may use for it. A trailing `...`
declares a **group**, any number of members `(:inputs, 1)`, `(:inputs, 2)`, …, which a graph
gives as a whole: a mixture's components, a sum's summands. A rule then targets any member,
`target = (:inputs, k)`, and reads a group as a tuple in member order. Names may contain
underscores: nothing in the package joins names together.

## Kinds

A [`Stochastic`](@ref) node has a density over its interfaces, `f(out | inputs)`. Its clusters
follow the graph's factorisation, it has an average energy, and, without groups, the macro also
defines its log-density as [`nodefunction`](@ref), which rule verification and the
[`NodeFunctionRuleFallback`](@ref) use.

A [`Deterministic`](@ref) node computes its output from its inputs, `out = f(inputs)`. Its
clusters are always its output and the joint over its inputs, whatever the factorisation. A rule
reaches the function through [`getnodefn`](@ref)`(ctx.node, Target(:out))`, which the engine
implements, since the engine's node owns the function and any static inputs folded into it.

```@docs
@define_factor_node
Stochastic
Deterministic
getnodefn
```

## What a node requires of a graph

A node may say what it requires of the graph it is placed in, and an engine checks it when it
creates the node, so a malformed graph is an error there rather than a rule silently reading
fewer components:

- `matched_groups = [(:m, :p)]`: groups with as many members as each other;
- `min_group_length = 2`: at least two members in every group; `0` allows an empty group;
- `factorisation = :meanfield`: only graphs giving every interface a cluster of its own, for a
  node whose rules are variational whatever the factorisation.

`static_inputs = :fold` asks the engine to fold inputs connected to constants and data into the
node's function, read through [`getnodefn`](@ref); such a node is built with its function.

## The declaration as data

The macro produces a [`NodeSpec`](@ref), which [`nodespec`](@ref) returns and every query below
reads. It shows itself as a summary:

```@repl nodes
using MessagePassingRulesBase # hide
struct Mixture end # hide
@define_factor_node(node = Mixture, type = Stochastic, interfaces = [:out, (:switch, aliases = [:s]), :inputs...], min_group_length = 2) # hide
MessagePassingRulesBase.nodespec(Mixture)
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
