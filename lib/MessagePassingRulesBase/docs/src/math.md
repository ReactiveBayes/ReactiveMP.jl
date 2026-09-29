```@meta
CurrentModule = MessagePassingRulesBase
```

# [Math helpers](@id math-helpers)

The linear algebra and the Gaussian algebra that rules of many nodes compute, kept here so that
every rule package computes them the same way. They are public, not exported: call them
qualified, `MessagePassingRulesBase.add_outer(V, m)`, or import them by name.

Each takes numbers as well as arrays, so a rule written once serves univariate and
multivariate inputs, and each keeps the float type of its inputs.

## Linear algebra

The functions ending in `!!` overwrite their argument where it is a dense `Array` and return a
new value otherwise, so they are given a value the caller owns, such as a fresh product.

```@docs
add_outer
trace_product
negate!!
scale!!
scaled_outer
diageye
promote_cluster
```

## Gaussians

Moments and energies of normal distributions, and what a normal factor sees of its parameters
under variational message passing.

```@docs
gaussian_second_moment
gaussian_cross_moment
gaussian_difference_moment
gaussian_average_energy
gaussian_variational_variance
gaussian_variational_covariance
gaussian_coupled_precision
gaussian_series_precision
```
