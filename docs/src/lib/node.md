
# [Nodes implementation](@id lib-node)

In message passing framework one of the most important concepts is factor node. 
Factor node represents a local function in a factorised representation of a generative model.

!!! note
    To quickly check the list of all available factor nodes that can be used in the model specification language call `?make_node` or `Base.doc(make_node)`.

## [Node traits](@id lib-node-traits)

Each factor node has to define [`as_node_functional_form`](@ref) trait function and to specify [`ValidNodeFunctionalForm`](@ref) singleton as a return object. By default [`as_node_functional_form`](@ref) returns [`UndefinedNodeFunctionalForm`](@ref). Objects that do not specify this property correctly cannot be used in model specification.

!!! note
    [`@node`](@ref) macro does that automatically

```@docs
ValidNodeFunctionalForm
UndefinedNodeFunctionalForm
as_node_functional_form
```

## [Node types](@id lib-node-types)

We distinguish different types of factor nodes to have a better control over Bethe Free Energy computation.
Each factor node has either [`Deterministic`](@ref) or [`Stochastic`](@ref) functional form type.

```@docs
Deterministic
Stochastic
isdeterministic
isstochastic
sdtype
```

```@setup lib-node-types
using ReactiveMP
```

For example `+` node has the [`Deterministic`](@ref) type:

```@example lib-node-types
plus_node = make_node(+)

println("Is `+` node deterministic: ", isdeterministic(plus_node))
println("Is `+` node stochastic: ", isstochastic(plus_node))
nothing #hide
```

On the other hand `Bernoulli` node has the [`Stochastic`](@ref) type:

```@example lib-node-types
bernoulli_node = make_node(Bernoulli)

println("Is `Bernoulli` node deterministic: ", isdeterministic(bernoulli_node))
println("Is `Bernoulli` node stochastic: ", isstochastic(bernoulli_node))
```

To get an actual instance of the type object we use [`sdtype`](@ref) function:

```@example lib-node-types
println("sdtype() of `+` node is ", sdtype(plus_node))
println("sdtype() of `Bernoulli` node is ", sdtype(bernoulli_node))
nothing #hide
```

## [Node functional dependencies pipeline](@id lib-node-functional-dependencies-pipeline)

```@docs
DefaultFunctionalDependencies
RequireMessageFunctionalDependencies
RequireMarginalFunctionalDependencies
RequireEverythingFunctionalDependencies
```

