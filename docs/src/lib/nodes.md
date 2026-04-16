# [Nodes implementation](@id lib-node)

In the message passing framework, one of the most important concepts is a factor node.
A factor node represents a local function in a factorised representation of a generative model.

```@docs
@node
ReactiveMP.FactorNode
ReactiveMP.FactorNodeLocalMarginal
```

## [Interfaces](@id lib-node-interfaces)

Every edge of a factor node — a connection to one variable — is represented by a [`ReactiveMP.NodeInterface`](@ref). When a `FactorNode` is constructed, one `NodeInterface` is created per edge. The constructor of `NodeInterface` immediately calls `ReactiveMP.create_new_stream_of_inbound_messages!` on the connected variable, which allocates a per-connection [`ReactiveMP.MessageObservable`](@ref) slot in the variable's `input_messages` and returns it. This observable is stored as `m_out` on the interface: it is the *outbound* message from the node's perspective (flowing toward the variable) and the *inbound* message from the variable's perspective.

At construction time all message streams are unconnected (lazy). The actual rule computations are wired up later during graph activation (see [Activation](@ref lib-node-activation)).

For nodes with a variable-length list of same-named edges (e.g. the `means` of a Gaussian Mixture node), [`ReactiveMP.IndexedNodeInterface`](@ref) wraps a `NodeInterface` and adds a positional index. The `ReactiveMP.ManyOf` container collects the corresponding streams for use in `@rule` dispatch; see the [Delta node](@ref lib-nodes-delta) documentation for usage examples.

```@docs
ReactiveMP.NodeInterface
ReactiveMP.IndexedNodeInterface
ReactiveMP.get_stream_of_inbound_messages
ReactiveMP.get_stream_of_outbound_messages
ReactiveMP.set_stream_of_outbound_messages!
ReactiveMP.tag
ReactiveMP.name
ReactiveMP.interfaces
ReactiveMP.getvariable
ReactiveMP.inputinterfaces
ReactiveMP.alias_interface
```

## [Activation](@id lib-node-activation)

Graph activation is the step that connects all lazy [`ReactiveMP.MessageObservable`](@ref) and [`ReactiveMP.MarginalObservable`](@ref) streams into a live reactive network. For factor nodes this is done by calling [`ReactiveMP.activate!`](@ref) with a [`ReactiveMP.FactorNodeActivationOptions`](@ref) that bundles all inference-time configuration.

```@docs
ReactiveMP.FactorNodeActivationOptions
ReactiveMP.activate!(::FactorNode, ::ReactiveMP.FactorNodeActivationOptions)
```

## [Adding a custom node](@id lib-custom-node)

`ReactiveMP.jl` exports the [`@node`](@ref) macro that allows for quick definition of a factor node with a __fixed__ number of edges. The example application can be the following:

```julia
struct MyNewCustomNode end

@node MyNewCustomNode   Stochastic         [ x, y, (z, aliases = [ d ] ) ]
#     ^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^      ^^^^^^^^^^^
#     Node's tag/name   Node's type        A fixed set of edges
#                       Another possible   The very first edge (in this example `x`) is considered
#                       value is           to be the output of the node
#                       `Deterministic`    - Edges can have aliases, e.g. `z` can be both `z` or `d`
```

This expression registers a new node that can be used with the inference engine. 
Note, however, that the `@node` macro does not generate any message passing update rules.
These must be defined using the [`@rule`](@ref) macro. 

## [Collecting node properties](@id lib-node-collect)

```@docs
ReactiveMP.collect_factorisation
ReactiveMP.collect_meta
ReactiveMP.default_meta
ReactiveMP.as_node_symbol
ReactiveMP.nodesymbol_to_nodefform
ReactiveMP.FunctionalDependencies
ReactiveMP.collect_functional_dependencies
```

## [Node types](@id lib-node-types)

We distinguish different types of factor nodes in order to have better control over Bethe Free Energy computation.
Each factor node has either the [`Deterministic`](@ref) or [`Stochastic`](@ref) functional form type.

```@docs
Deterministic
Stochastic
isdeterministic
isstochastic
sdtype
```

```@setup lib-node-types
using ReactiveMP, BayesBase, Distributions, ExponentialFamily
```

For example the `+` node has the [`Deterministic`](@ref) type:

```@example lib-node-types
println("Is `+` node deterministic: ", isdeterministic(sdtype(+)))
println("Is `+` node stochastic: ", isstochastic(sdtype(+)))
nothing #hide
```

On the other hand, the `Bernoulli` node has the [`Stochastic`](@ref) type:

```@example lib-node-types
println("Is `Bernoulli` node deterministic: ", isdeterministic(sdtype(Bernoulli)))
println("Is `Bernoulli` node stochastic: ", isstochastic(sdtype(Bernoulli)))
nothing #hide
```

To get an actual instance of the type object we use [`sdtype`](@ref) function:

```@example lib-node-types
println("sdtype() of `+` node is ", sdtype(+))
println("sdtype() of `Bernoulli` node is ", sdtype(Bernoulli))
nothing #hide
```

## [Node functional dependencies](@id lib-node-functional-dependencies)

The generic implementation of factor nodes in ReactiveMP supports custom functional dependencies policies. Briefly, the __functional dependencies__ define what
dependencies are needed to compute a single message. As an example, consider the belief-propagation message update equation for a factor node $f$ with three edges: $x$, $y$ and $z$:

```math
\mu(x) = \int \mu(y) \mu(z) f(x, y, z) \mathrm{d}y \mathrm{d}z
```

Here we see that in the standard setting for the belief-propagation message out of edge $x$, we need only messages from the edges $y$ and $z$. In contrast, consider the variational message update rule equation with mean-field assumption:

```math
\mu(x) = \exp \int q(y) q(z) \log f(x, y, z) \mathrm{d}y \mathrm{d}z
```

We see that in this setting, we do not need messages $\mu(y)$ and $\mu(z)$, but only the marginals $q(y)$ and $q(z)$. 

## [List of functional dependencies policies](@id lib-node-functional-dependencies-policies)

The purpose of a __functional dependencies__ policy is to determine functional dependencies (a set of messages or marginals) that are needed to compute a single message. By default, `ReactiveMP.jl` uses so-called `DefaultFunctionalDependencies` that correctly implements belief-propagation and variational message passing schemes (including both mean-field and structured factorisations). The full list of built-in policies is presented below:

```@docs
ReactiveMP.DefaultFunctionalDependencies
ReactiveMP.RequireMessageFunctionalDependencies
ReactiveMP.RequireMarginalFunctionalDependencies
ReactiveMP.RequireEverythingFunctionalDependencies
```

## [Node traits](@id lib-node-traits)

Each factor node has to define the [`ReactiveMP.is_predefined_node`](@ref) trait function and to specify a [`ReactiveMP.PredefinedNodeFunctionalForm`](@ref) 
singleton as a return object. By default [`ReactiveMP.is_predefined_node`](@ref) returns [`ReactiveMP.UndefinedNodeFunctionalForm`](@ref). 
Objects that do not specify this property correctly cannot be used in model specification.

!!! note
    `@node` macro does that automatically

```@docs
ReactiveMP.PredefinedNodeFunctionalForm
ReactiveMP.UndefinedNodeFunctionalForm
ReactiveMP.is_predefined_node
```

## [Stream postprocessors](@id lib-node-stream-postprocessors)

Stream postprocessors are composable transformations applied to the reactive observables produced during activation — outbound message streams, marginal streams, and score streams. They are attached to a node via [`ReactiveMP.FactorNodeActivationOptions`](@ref) and to a random variable via [`ReactiveMP.RandomVariableActivationOptions`](@ref), and can be used for scheduling or custom instrumentation.

See the dedicated [Stream postprocessors](@ref lib-stream-postprocessors) page for a full description and API reference.

## [List of predefined factor node](@id lib-predefined-nodes)    

To quickly check the list of all predefined factor nodes, call `?ReactiveMP.is_predefined_node` or `Base.doc(ReactiveMP.is_predefined_node)`.

```
?ReactiveMP.is_predefined_node
```

```@eval
using ReactiveMP, Markdown
Markdown.parse(string(Base.doc(Base.Docs.Binding(ReactiveMP, :is_predefined_node))))
```
