
# [Nodes implementation](@id lib-node)

In the message passing framework, one of the most important concepts is a factor node.
A factor node represents a local function in a factorised representation of a generative model.

```@docs
@node
ReactiveMP.FactorNode
ReactiveMP.FactorNodeLocalMarginal
ReactiveMP.NodeInterface
ReactiveMP.IndexedNodeInterface
ReactiveMP.messagein
ReactiveMP.messageout
ReactiveMP.tag
ReactiveMP.name
ReactiveMP.interfaces
ReactiveMP.getvariable
ReactiveMP.inputinterfaces
ReactiveMP.alias_interface
ReactiveMP.collect_factorisation
ReactiveMP.collect_pipeline
ReactiveMP.collect_meta
ReactiveMP.default_meta
ReactiveMP.as_node_symbol
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

## [Node functional dependencies pipeline](@id lib-node-functional-dependencies-pipeline)

The generic implementation of factor nodes in ReactiveMP supports custom functional dependency pipelines. Briefly, the __functional dependencies pipeline__ defines what
dependencies are need to compute a single message. As an example, consider the belief-propagation message update equation for a factor node $f$ with three edges: $x$, $y$ and $z$:

```math
\mu(x) = \int \mu(y) \mu(z) f(x, y, z) \mathrm{d}y \mathrm{d}z
```

Here we see that in the standard setting for the belief-propagation message out of edge $x$, we need only messages from the edges $y$ and $z$. In contrast, consider the variational message update rule equation with mean-field assumption:

```math
\mu(x) = \exp \int q(y) q(z) \log f(x, y, z) \mathrm{d}y \mathrm{d}z
```

We see that in this setting, we do not need messages $\mu(y)$ and $\mu(z)$, but only the marginals $q(y)$ and $q(z)$. The purpose of a __functional dependencies pipeline__ is to determine functional dependencies (a set of messages or marginals) that are needed to compute a single message. By default, `ReactiveMP.jl` uses so-called `DefaultFunctionalDependencies` that correctly implements belief-propagation and variational message passing schemes (including both mean-field and structured factorisations). The full list of built-in pipelines is presented below:

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

## [Node pipelines](@id lib-node-pipelines)

```@docs
ReactiveMP.AbstractPipelineStage
ReactiveMP.apply_pipeline_stage
ReactiveMP.EmptyPipelineStage
ReactiveMP.CompositePipelineStage
ReactiveMP.LoggerPipelineStage
ReactiveMP.DiscontinuePipelineStage
ReactiveMP.AsyncPipelineStage
ReactiveMP.ScheduleOnPipelineStage
ReactiveMP.schedule_updates
```

## [List of predefined factor node](@id lib-predefined-nodes)    

To quickly check the list of all predefined factor nodes, call `?ReactiveMP.is_predefined_node` or `Base.doc(ReactiveMP.is_predefined_node)`.

```
?ReactiveMP.is_predefined_node
```

```@eval
using ReactiveMP, Markdown
Markdown.parse(string(Base.doc(Base.Docs.Binding(ReactiveMP, :is_predefined_node))))
```
