# [Flow models](@id flow-models)

A flow model is an invertible map ``f = f_L \circ \dots \circ f_1`` of ``\mathbb{R}^d`` onto
itself, each layer ``f_l`` invertible with a Jacobian that is cheap to compute. It is built in
two steps: [`FlowModel`](@ref) sizes the layers for the dimension ``d``, and [`compile`](@ref)
sets their parameters, explicit ones or drawn from a generator, giving the
[`CompiledFlowModel`](@ref) the [`Flow`](@ref) node takes.

```jldoctest; setup = :(using FlowMessagePassingRules)
julia> model = FlowModel(3, (AdditiveCouplingLayer(RadialFlow()), AdditiveCouplingLayer(PlanarFlow())));

julia> nr_params(model)   # two radial flows of 3 parameters, two planar flows of 3
12

julia> compiled = compile(model);

julia> x = [0.1, -0.4, 0.7];

julia> FlowMessagePassingRules.backward(compiled, FlowMessagePassingRules.forward(compiled, x)) ≈ x
true
```

## Models

```@docs
FlowModel
CompiledFlowModel
compile(::FlowModel)
compile(::FlowModel, ::Vector)
nr_params
getlayers
Base.eltype(::CompiledFlowModel)
```

## Layers

A model's tuple of layers may start with an [`InputLayer`](@ref) in place of the dimension. An
[`AdditiveCouplingLayer`](@ref) holds the coupling flows and, by default, is followed by a random
[`PermutationLayer`](@ref), so that the next coupling layer mixes the coordinates in another
order.

!!! warning "Scalar partitions only"
    An [`AdditiveCouplingLayer`](@ref) works only with `partition_dim = 1`, its default: a larger
    partition builds a model that throws when it is evaluated.

```@docs
InputLayer
AdditiveCouplingLayer
PermutationLayer
```

## Coupling flows

The functions an [`AdditiveCouplingLayer`](@ref) adds, one per coordinate after the first. Given
without arguments they are placeholders, sized by the model and given parameters by
[`compile`](@ref).

```@docs
PlanarFlow
RadialFlow
```

## Evaluating a model

The functions the rules call on a compiled model are public but not exported, and are called
qualified, `FlowMessagePassingRules.forward(model, x)`. Each broadcasts over a vector of points
with the model fixed.

```@docs
FlowMessagePassingRules.forward
FlowMessagePassingRules.backward
FlowMessagePassingRules.jacobian
FlowMessagePassingRules.inv_jacobian
FlowMessagePassingRules.forward_jacobian
FlowMessagePassingRules.backward_inv_jacobian
```

## Permutations

```@docs
PermutationMatrix
```
