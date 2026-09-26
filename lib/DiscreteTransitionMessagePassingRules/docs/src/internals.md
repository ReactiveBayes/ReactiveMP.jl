# Internals

The tensor algebra of the rules: multiplying a tensor by a vector or a tensor along some of its
axes, and summing those axes out. The rules of [`DiscreteTransition`](@ref) are built from these
two; they are not exported.

```@docs
DiscreteTransitionMessagePassingRules.multiply_dimensions!
DiscreteTransitionMessagePassingRules.sum_out_dimensions
```
