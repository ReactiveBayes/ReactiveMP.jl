# [Factor nodes](@id lib-node)

A factor node is one local function of a factorised model. The engine creates a
[`FactorNode`](@ref) for each factor, connects it to its variables through interfaces, and, when
the graph is activated, wires every outbound message to the update rule that computes it.

The engine defines no node. A node is declared with
[`@define_factor_node`](@extref MessagePassingRulesBase.@define_factor_node), and its rules with
the rule macros of [`MessagePassingRulesBase`](https://reactivebayes.github.io/MessagePassingRulesBase.jl/dev/);
the standard nodes are those of
[`StandardMessagePassingRules`](https://reactivebayes.github.io/StandardMessagePassingRules.jl/dev/),
and the others have packages of their own ([The ecosystem](@ref ecosystem)).

## [Creating a node](@id lib-node-create)

A node is created from its node type, the variables it connects and a factorisation. The
interfaces are `(name, variable)` pairs, or `((group, k), variable)` for the members of an
interface group, and the factorisation is a tuple of clusters, each a tuple of those keys:

```julia
x, y, v = randomvar(), randomvar(), constvar(1.0)
node = factornode(NormalMeanVariance, [(:out, y), (:μ, x), (:v, v)], ((:out, :μ), (:v,)))
```

The engine puts interfaces and clusters in declaration order, whatever order they are given in.
A joint cluster's local marginal is keyed by its member tuple, `(:out, :μ)`. A cluster may hold a
whole group, keyed by its name, `(:in,)`, or some of its members, each keyed with its index,
`(:out, (:in, 1))`. Without a factorisation, a node has one cluster over every interface, and its
rules are those of belief propagation; with one cluster per interface, `((:out,), (:μ,), (:v,))`,
those of mean-field variational message passing.

```@docs
FactorNode
factornode
functionalform
getinterfaces
ReactiveMP.FactorNodeLocalMarginal
```

## [Interfaces](@id lib-node-interfaces)

Every edge of a factor node, a connection to one variable, is a [`ReactiveMP.NodeInterface`](@ref).
Creating it allocates a slot for the node's messages in the variable's inbound messages: the
interface's outbound message is the variable's inbound one, and the interface's inbound message
the variable's outbound one. The members of an interface group, such as the components of a
mixture, are [`ReactiveMP.IndexedNodeInterface`](@ref)s, which add the member's index.

```@docs
ReactiveMP.NodeInterface
ReactiveMP.IndexedNodeInterface
ReactiveMP.name
ReactiveMP.getvariable
ReactiveMP.get_stream_of_outbound_messages
ReactiveMP.get_stream_of_inbound_messages
ReactiveMP.set_stream_of_outbound_messages!
```

## [Activation](@id lib-node-activation)

Activation connects the lazy message and marginal streams into a live network, after the
node's variables are activated. For each interface on a random or a data variable, the engine
finds the inputs its rule needs, from the dependencies the node's algorithm declares or from the
default scheme, and subscribes to them in declaration order, which in variational message
passing is the update schedule. What it runs with, the algorithm, callbacks, annotations,
diagnostics, services, a rule fallback and log scales, is the node's
[`ReactiveMP.FactorNodeActivationOptions`](@ref), described on
[Activation options](@ref lib-activation-options).

```@docs
ReactiveMP.activate!(::FactorNode, ::ReactiveMP.FactorNodeActivationOptions)
```

How a node's inputs are chosen and labelled is on the [Internals](@ref internals-dependencies)
page.

## [Static inputs](@id lib-node-static-inputs)

A deterministic node declared with `static_inputs = :fold`, such as the Delta node, folds the
inputs whose values are known, constants and data, into its node function, so that its rules see
only the random ones. Such a node needs its function at creation,
`factornode(f, …; nodefn = f)`, and its rules reach it with
[`getnodefn`](@extref MessagePassingRulesBase.getnodefn). See
[`ReactiveMP.StaticFold`](@ref) on the [Internals](@ref internals-static-inputs) page.

## [Node kinds](@id lib-node-types)

Each factor node is either deterministic or stochastic. The kind decides how the node enters the
free energy: a deterministic node has no average energy, and its clusters are always its output
and the joint over its inputs.

```@docs
sdtype
isdeterministic
isstochastic
```

```@setup lib-node-types
using ReactiveMP, StandardMessagePassingRules, Distributions
```

The `+` node is deterministic, and the `Bernoulli` node stochastic:

```@example lib-node-types
isdeterministic(sdtype(+)), isstochastic(sdtype(Bernoulli))
```
