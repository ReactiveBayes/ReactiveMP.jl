# [Factor nodes](@id lib-node)

A factor node represents one local function of a factorised generative model. The engine
creates a [`FactorNode`](@ref) for each factor of a model, connects it to its variables through
interfaces, and, when the graph is activated, wires every outbound message to the update rule
that computes it.

The engine does not define any node itself. A node is declared with `@define_factor_node` from
`MessagePassingRulesBase`, and its rules with that package's rule macros; see
[Defining nodes and rules](@ref rules-defining). The standard nodes are declared in
`StandardMessagePassingRules` ([Standard rules](@ref packages-standard)).

```@docs
FactorNode
factornode
functionalform
getinterfaces
ReactiveMP.FactorNodeLocalClusters
ReactiveMP.FactorNodeLocalMarginal
ReactiveMP.clusterkey
```

A node is created from its functional form, the variables it connects, and a factorisation.
The interfaces are `(name, variable)` pairs, or `((group, k), variable)` for the members of an
interface group, and the factorisation is a tuple of clusters, each a tuple of those keys:

```julia
x, y, v = randomvar(), randomvar(), constvar(1.0)
node = factornode(NormalMeanVariance, [(:out, y), (:μ, x), (:v, v)], ((:out, :μ), (:v,)))
```

The engine puts interfaces and clusters in declaration order, whatever order they are given in.
A joint cluster's local marginal is keyed by its member tuple, `(:out, :μ)`, and a cluster may
hold a whole group, `(:in,)`, but not only some of its members.

## [Interfaces](@id lib-node-interfaces)

Every edge of a factor node, a connection to one variable, is a [`ReactiveMP.NodeInterface`](@ref).
Creating it allocates a slot for the node's message in the variable's inbound messages: the
interface's outbound message is the variable's inbound one. All streams are lazy until the graph
is activated. The members of an interface group, such as the components of a mixture, are
[`ReactiveMP.IndexedNodeInterface`](@ref)s, which add the member's index.

```@docs
ReactiveMP.NodeInterface
ReactiveMP.IndexedNodeInterface
ReactiveMP.get_stream_of_inbound_messages
ReactiveMP.get_stream_of_outbound_messages
ReactiveMP.set_stream_of_outbound_messages!
ReactiveMP.tag
ReactiveMP.name
ReactiveMP.getvariable
```

## [Activation](@id lib-node-activation)

Activation connects the lazy message and marginal streams into a live reactive network. For a
factor node it is [`ReactiveMP.activate!`](@ref) with a [`ReactiveMP.FactorNodeActivationOptions`](@ref),
which carries the algorithm the node's rules run under, a stream postprocessor, annotation
processors and callbacks. For each interface, the engine finds the inputs its rule needs, from
the node's declared dependencies or from the default scheme, and subscribes to them in
declaration order, which is the update schedule. See
[Algorithms and dependencies](@ref rules-algorithms) for how a node chooses its inputs.

```@docs
ReactiveMP.FactorNodeActivationOptions
ReactiveMP.getalgorithm
ReactiveMP.activate!(::FactorNode, ::ReactiveMP.FactorNodeActivationOptions)
ReactiveMP.default_dependencies
ReactiveMP.declared_dependencies
ReactiveMP.extended_default_dependencies
ReactiveMP.rule_target
ReactiveMP.input_label
ReactiveMP.GroupMember
ReactiveMP.GroupInputs
ReactiveMP.input_names
```

## [Static inputs](@id lib-node-static-inputs)

A deterministic node declared with `static_inputs = :fold`, such as the Delta node, folds the
inputs whose values are known (constants and data) into its node function, so its rules only
see the random ones. Such a node needs its function at creation, `factornode(f, …; nodefn = f)`.

```@docs
ReactiveMP.StaticFold
ReactiveMP.with_statics
```

## [Node types](@id lib-node-types)

Each factor node is either deterministic or stochastic. The distinction decides how a node's
contribution to the free energy is computed: a deterministic node's clusters are always its
output and the joint over its inputs.

```@docs
isdeterministic
isstochastic
sdtype
```

```@setup lib-node-types
using ReactiveMP, StandardMessagePassingRules, BayesBase, Distributions, ExponentialFamily
```

The `+` node is deterministic, and the `Bernoulli` node stochastic:

```@example lib-node-types
isdeterministic(sdtype(+)), isstochastic(sdtype(Bernoulli))
```

## [Stream postprocessors](@id lib-node-stream-postprocessors)

Stream postprocessors are composable transformations of the streams activation creates:
outbound messages, marginals and scores. They are given to a node through
[`ReactiveMP.FactorNodeActivationOptions`](@ref) and to a random variable through
[`ReactiveMP.RandomVariableActivationOptions`](@ref); see
[Stream postprocessors](@ref lib-stream-postprocessors).
