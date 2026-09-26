# Gauss–Hermite cubature

Expectations under a normal, by a tensor-product Gauss–Hermite rule: `E[f(x)] ≈ Σ wᵢ f(xᵢ)` over
the points [`getpoints`](@ref) and weights [`getweights`](@ref) of the normal. On top of them,
[`approximate_meancov`](@ref) gives the moments of a normal reweighted by a function, the
building block of projecting a non-Gaussian update back onto a normal.

```@docs
GaussHermiteCubature
ghcubature
getpoints
getweights
approximate_meancov
```
