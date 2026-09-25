# [Flow](@id packages-flow)

```@docs
FlowMessagePassingRules
```

`Flow` is the deterministic node `out = f(in)` for an invertible flow model `f`, built from
layers and compiled with its parameters. A normal is pushed through the model by linearisation,
with the model's own Jacobians, or by the unscented transform, under `FlowApproximation`, which
the model must give.

```@docs
Flow
FlowApproximation
```

## [Flow models](@id packages-flow-models)

A model is a tuple of layers. Compiling it gives it parameters: explicit ones, or drawn from a
generator, the task's unless one is given.

```@docs
FlowModel
CompiledFlowModel
compile
nr_params
getlayers
InputLayer
AdditiveCouplingLayer
PermutationLayer
PlanarFlow
RadialFlow
PermutationMatrix
```

The functions the rules use are public, and called qualified, `FlowMessagePassingRules.forward`:

```@docs
FlowMessagePassingRules.forward
FlowMessagePassingRules.backward
FlowMessagePassingRules.jacobian
FlowMessagePassingRules.inv_jacobian
FlowMessagePassingRules.forward_jacobian
FlowMessagePassingRules.backward_inv_jacobian
Base.eltype(::CompiledFlowModel)
```

`compile(model)`, `PermutationMatrix(dim)` and `PermutationLayer()` draw random parameters and
permutations. Each takes a generator as its first argument, `compile(rng, model)`, and without
one draws from the task's. The node's rules draw nothing.
