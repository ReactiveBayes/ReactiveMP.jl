```@meta
CurrentModule = MessagePassingRulesBase
```

# [Math helpers](@id math-helpers)

The math helpers are the linear algebra and the Gaussian algebra that the rules of many nodes
compute. They live here so that every rule package computes them the same way.

They are public but not exported. Call them qualified, as
`MessagePassingRulesBase.add_outer(V, m)`, or import them by name.

Each helper takes numbers as well as arrays, so a rule written once serves univariate and
multivariate inputs. Each keeps the float type of its inputs.

## Linear algebra

The functions ending in `!!` overwrite their argument when it is a dense `Array`, and return a
new value otherwise. Give them only a value you own, such as a fresh product.

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

These helpers compute the moments and energies of normal distributions. They also compute what
a normal factor sees of its parameters under
[variational message passing](@ref glossary-vmp).

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
