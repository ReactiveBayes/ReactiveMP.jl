# Gauss–Hermite cubature

Gauss–Hermite cubature approximates an expectation under a normal by a weighted sum of the
function at fixed points. It suits expectations that have no closed form, in one or a few
dimensions.

## The method

For a scalar normal ``N(m, v)``, the substitution ``x = m + \sqrt{2v}\, ξ`` turns the
expectation into an integral against ``e^{-ξ^2}``, which the Gauss–Hermite rule with `p` nodes
``ξ_i`` and weights ``w_i`` evaluates:

```math
\mathrm{E}[f(x)] = \frac{1}{\sqrt{π}} \int f\bigl(m + \sqrt{2v}\, ξ\bigr) e^{-ξ^2} \, \mathrm{d}ξ
≈ \frac{1}{\sqrt{π}} \sum_{i=1}^{p} w_i f\bigl(m + \sqrt{2v}\, ξ_i\bigr).
```

The nodes are the roots of the Hermite polynomial of degree `p`, and the sum is exact when `f`
is a polynomial of degree up to `2p - 1`. In `d` dimensions the points form a tensor-product grid
of `p^d` points, mapped through a square root of the covariance, so the method stays practical
for a handful of dimensions only.

[`getpoints`](@ref) and [`getweights`](@ref) return the mapped points and the normalised weights,
``w_i / \sqrt{π}``, which sum to one. Three points integrate ``x^2`` exactly, so they recover
``\mathrm{E}[x^2] = m^2 + v``:

```@example gauss-hermite
using MessagePassingRulesApproximations

gh = ghcubature(3)
m, v = 1.0, 4.0
points, weights = collect(getpoints(gh, m, v)), collect(getweights(gh, m, v))
sum(weights .* points .^ 2), m^2 + v
```

## Moments of a reweighted normal

[`approximate_meancov`](@ref) returns the mean and variance of the density proportional to
``g(x) N(x \mid m, v)``, for a non-negative ``g``. A normal prior multiplied by a likelihood
``g`` has this form, and its moments define the normal closest to the product. For a standard
normal prior and a logistic likelihood of a positive label:

```@example gauss-hermite
likelihood(x) = 1 / (1 + exp(-4x))
approximate_meancov(ghcubature(20), likelihood, 0.0, 1.0)
```

The likelihood favours positive values, so the mean moves above zero and the variance shrinks
below one.

## API

```@docs
GaussHermiteCubature
ghcubature
getpoints
getweights
approximate_meancov
```
