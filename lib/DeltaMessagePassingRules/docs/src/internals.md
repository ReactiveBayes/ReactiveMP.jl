# [Internals](@id delta-internals)

For contributors. None of these names is public.

The two forms of [`DeltaApproximation`](@ref), which the dependencies and rules dispatch on:

```@docs
DeltaMessagePassingRules.UnknownInverse
DeltaMessagePassingRules.KnownInverse
```

The routines the Gaussian rules share: the two methods differ only in these.

```@docs
DeltaMessagePassingRules.approximate_normal
DeltaMessagePassingRules.forward_statistics
```

The error of a method the node does not accept, and the projection families of
[`CVIProjection`](@ref):

```@docs
DeltaMessagePassingRules.delta_method_hint
DeltaMessagePassingRules.get_kth_in_form
```
