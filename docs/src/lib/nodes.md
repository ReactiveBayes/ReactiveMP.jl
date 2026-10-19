# [Factor nodes](@id lib-node)

A [factor node](@extref MessagePassingRulesBase glossary-factor-node) is one factor of a
factorised model. The engine creates a [`FactorNode`](@ref) for each factor and connects it to its
variables through [interfaces](@extref MessagePassingRulesBase glossary-interface). When you
activate the graph, the engine wires each of the node's outbound messages to the
[rule](@extref MessagePassingRulesBase glossary-rule) that computes it.

The engine defines no node. You declare a node with
[`@define_factor_node`](@extref MessagePassingRulesBase.@define_factor_node) and its rules with
the other macros of [MessagePassingRulesBase](https://reactivebayes.github.io/MessagePassingRulesBase.jl/dev/).
The standard nodes, the distributions and the arithmetic, come from
[StandardMessagePassingRules](@extref StandardMessagePassingRules StandardMessagePassingRules),
and the others from packages of their own ([The ecosystem](@ref ecosystem)).

```@setup nodes
using ReactiveMP, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, Rocket
import ReactiveMP: activate!, FactorNodeActivationOptions, get_stream_of_marginals
include(joinpath(pkgdir(ReactiveMP), "docs", "nodes.jl"))
```

The examples use the `Gaussian` node of [The example node](@ref example-node).

## [Creating a node](@id lib-node-create)

[`factornode`](@ref) takes the node type, the variables it connects and a
[factorisation](@extref MessagePassingRulesBase glossary-factorisation). Each interface is a
`(name, variable)` pair. The factorisation is a tuple of
[clusters](@extref MessagePassingRulesBase glossary-cluster), each a tuple of interface names:

```@example nodes
x, y = randomvar(label = :x), randomvar(label = :y)
node = factornode(Gaussian, [(:v, constvar(1.0)), (:out, y), (:μ, x)], ((:out, :μ), (:v,)))
```

The node draws its interfaces, the variables they connect to and its clusters. The engine puts
interfaces and clusters in declaration order, whatever order you give them in.

Without a factorisation, a node has one cluster over every interface, and its rules are those of
[belief propagation](@extref MessagePassingRulesBase glossary-belief-propagation). With one
cluster per interface, `((:out,), (:μ,), (:v,))`, they are those of
[mean-field](@extref MessagePassingRulesBase glossary-mean-field) variational message passing.
Every interface belongs to exactly one cluster.

Each cluster has a local marginal, keyed by its members. A joint cluster's key is its member
tuple, `(:out, :μ)`:

```@example nodes
ReactiveMP.get_node_local_marginals(ReactiveMP.getlocalclusters(node))
```

A node with a [group](@extref MessagePassingRulesBase glossary-group), an interface with any
number of members, takes each member as `((group, k), variable)`. A cluster may hold the whole
group, keyed by its name, `(:in,)`, or some of its members, each keyed with its index,
`(:out, (:in, 1))`. The [static inputs](@ref lib-node-static-inputs) example below creates a node
with a group.

```@docs
ReactiveMP.AbstractFactorNode
FactorNode
factornode
functionalform
getinterfaces
ReactiveMP.getinterface
ReactiveMP.getlocalclusters
ReactiveMP.get_node_local_marginals
ReactiveMP.FactorNodeLocalMarginal
```

## [Interfaces](@id lib-node-interfaces)

Every edge of a factor node, its connection to one variable, is a
[`ReactiveMP.NodeInterface`](@ref):

```@example nodes
getinterfaces(node)
```

Creating an interface allocates a stream for the node's messages among the variable's inbound
messages. So the interface's outbound message is the variable's inbound one, and the interface's
inbound message is the variable's outbound one. The members of a group are
[`ReactiveMP.IndexedNodeInterface`](@ref)s, which add the member's index.

```@example nodes
interface = ReactiveMP.getinterface(node, 3)
ReactiveMP.name(interface), ReactiveMP.getvariable(interface)
```

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

Activation connects the node's lazy message and marginal streams into a live network. You
activate a node after its variables. For each interface on a random or a data variable, the
engine finds the inputs its rule needs, from the
[dependencies](@extref MessagePassingRulesBase glossary-dependencies) the node's algorithm
declares or from the [default scheme](@extref MessagePassingRulesBase glossary-default-scheme).
It subscribes to them in declaration order, which in variational message passing is the update
schedule.

```@example nodes
x, y = randomvar(label = :x), randomvar(label = :y)
prior = factornode(Gaussian, [(:out, x), (:μ, constvar(0.0)), (:v, constvar(10.0))])
likelihood = factornode(Gaussian, [(:out, y), (:μ, x), (:v, constvar(1.0))])
foreach(v -> activate!(v, RandomVariableActivationOptions()), (x, y))
foreach(n -> activate!(n, FactorNodeActivationOptions()), (prior, likelihood))

subscription = subscribe!(get_stream_of_marginals(y), (q) -> println("q(y) = ", q))
nothing # hide
```

The variables are new ones, since a variable waits for a message from every node connected to
it. Here `y` is latent, and its marginal is the prior predictive, ``\mathcal{N}(0, 10 + 1)``. The
node runs with its [`ReactiveMP.FactorNodeActivationOptions`](@ref): the algorithm, callbacks,
annotations, diagnostics, services, a rule fallback and log scales, described on
[Activation options](@ref lib-activation-options).

```@docs
ReactiveMP.activate!(::FactorNode, ::ReactiveMP.FactorNodeActivationOptions)
```

The [Internals](@ref internals-dependencies) page describes how the engine chooses and labels a
node's inputs.

## [Static inputs](@id lib-node-static-inputs)

A deterministic node declared with `static_inputs = :fold` folds the inputs whose values are
known, constants and data, into its node function. Its rules see only the random inputs. You give
the function when you create the node, `factornode(…; nodefn = f)`, and a rule reaches it with
[`getnodefn`](@extref MessagePassingRulesBase.getnodefn). The Delta node of
[DeltaMessagePassingRules](@extref DeltaMessagePassingRules DeltaMessagePassingRules) works this
way.

The node below computes `out = f(ins...)` for an affine `f`. Its rule pushes a normal message
through the folded function:

```@example nodes
struct Affine end

@define_factor_node(node = Affine, type = Deterministic, interfaces = [:out, :ins...], static_inputs = :fold)

@define_message_update_rule(
    node = Affine, target = :out, args = (m[:ins...]::NormalMeanVariance,), ctx = (:node,),
    body = (ctx, args) -> begin
        f = getnodefn(ctx.node, MessagePassingRulesBase.Target(:out))
        x = only(args.m[:ins])
        slope = f(mean(x) + 1) - f(mean(x))
        NormalMeanVariance(f(mean(x)), slope^2 * var(x))
    end,
)

a, b = randomvar(label = :a), randomvar(label = :b)
source = factornode(Gaussian, [(:out, a), (:μ, constvar(0.0)), (:v, constvar(1.0))])
affine = factornode(Affine, [(:out, b), ((:ins, 1), a), ((:ins, 2), constvar(3.0))]; nodefn = (a, k) -> k * a + 1)
```

The constant `3.0` is folded into the function, so the node shows one member of `ins`, and the
rule sees one input. The function it reaches is `a -> 3.0 * a + 1`, and it sends
``\mathcal{N}(3 \cdot 0 + 1, 3^2 \cdot 1)`` towards `b`:

```@example nodes
foreach(v -> activate!(v, RandomVariableActivationOptions()), (a, b))
foreach(n -> activate!(n, FactorNodeActivationOptions()), (source, affine))
affine_subscription = subscribe!(get_stream_of_marginals(b), (q) -> println("q(b) = ", q))
nothing # hide
```

The [Internals](@ref internals-static-inputs) page describes [`ReactiveMP.StaticFold`](@ref), which
holds the folded function.

## [Node kinds](@id lib-node-types)

Each factor node is either deterministic or stochastic. A
[stochastic node](@extref MessagePassingRulesBase glossary-stochastic-node) is a density over its
interfaces. A [deterministic node](@extref MessagePassingRulesBase glossary-deterministic-node)
computes its output as a function of its inputs. The kind decides how the node enters the
[free energy](@ref lib-score): a deterministic node has no average energy, and its clusters are
always its output and the joint over its inputs.

```@example nodes
isstochastic(sdtype(Gaussian)), isdeterministic(sdtype(Affine))
```

```@docs
sdtype
isdeterministic
isstochastic
```
