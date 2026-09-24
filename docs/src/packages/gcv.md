# [GCV](@id packages-gcv)

```@docs
GCVMessagePassingRules
```

The Gaussian controlled variance node, `y ~ N(x, exp(κz + ω))`, lets the variance of `y` about `x`
depend on other variables, as in hierarchical Gaussian filters. Its rules are variational under
`q(y, x) q(z) q(κ) q(ω)` or mean-field. The messages towards `z`, `κ` and `ω` are
[`ExponentialLinearQuadratic`](@ref) densities whose moments a cubature computes, the method of
[`GCVApproximation`](@ref). The package also gives `NormalMeanVariance` and `NormalMeanPrecision`
rules for such a message on their `out`, which is how a GCV's `y` is usually the mean of another
normal.

```@docs
GCV
GCVApproximation
ExponentialLinearQuadratic
```
