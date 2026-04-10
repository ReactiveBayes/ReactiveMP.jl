# [Flow node](@id lib-nodes-flow)

The flow node encodes a **normalizing flow** — a parameterized, invertible transformation that maps a simple base distribution (e.g., a Gaussian) into a complex, multimodal one. Because the transformation is invertible, both forward and backward messages can be computed without approximation.

```julia
y ~ Flow(x) where { meta = FlowMeta(compiled_model) }
```

This asserts that `y = f(x)` where `f` is the composed invertible transformation defined by the flow model. The node type is [`Deterministic`](@ref).

## [Building a flow model](@id lib-nodes-flow-model)

A flow is assembled from **layers** stacked inside a [`FlowModel`](@ref). Each layer is an invertible mapping. ReactiveMP.jl provides:

| Layer | Description |
|-------|-------------|
| [`PlanarFlow`](@ref) | Planar contraction/expansion along a learned direction |
| [`RadialFlow`](@ref) | Radial contraction/expansion around a learned center point |
| [`AdditiveCouplingLayer`](@ref) | Affine coupling — splits the input and transforms one half conditioned on the other |
| [`PermutationLayer`](@ref) | Permutes the input dimensions to mix information across coupling layers |
| [`InputLayer`](@ref) | Declares the input dimensionality; must be the first layer in a model |

Layers are composed into a `FlowModel` and then **compiled** into a [`CompiledFlowModel`](@ref) before use. Compilation fixes the layer sizes and randomly initializes parameters:

```julia
model = FlowModel((
    InputLayer(2),
    AdditiveCouplingLayer(PlanarFlow()),
    PermutationLayer(),
    AdditiveCouplingLayer(PlanarFlow()),
))

compiled = compile(model)   # randomly initialized parameters

# or pass your own parameter vector:
compiled = compile(model, params)
```

The compiled model is then wrapped in [`FlowMeta`](@ref) and attached to the node:

```julia
y ~ Flow(x) where { meta = FlowMeta(compiled) }
```

## [Approximation inside the flow node](@id lib-nodes-flow-approximation)

By default, [`FlowMeta`](@ref) uses [`Linearization`](@ref) for any messages that require approximation (e.g., backward messages when the inverse Jacobian is unavailable in closed form). A different approximation can be passed as the second argument:

```julia
FlowMeta(compiled, Unscented())
```

## [Learning flow parameters](@id lib-nodes-flow-learning)

!!! note
    See also the [Flow tutorial](https://reactivebayes.github.io/RxInfer.jl/stable/examples/overview/) in the RxInfer.jl documentation for a complete end-to-end example.

```@docs
PlanarFlow
RadialFlow
FlowModel
CompiledFlowModel
compile
AdditiveCouplingLayer
InputLayer
PermutationLayer
FlowMeta
```
