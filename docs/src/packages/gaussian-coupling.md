# [GaussianCoupling](@id packages-gaussian-coupling)

```@docs
GaussianCouplingMessagePassingRules
```

`GaussianCoupling` couples two variables by `exp(out ⋅ a ⋅ in)`, with a constant coefficient `a`.
With `NormalWeightedMeanPrecision(b_i, A_ii)` priors and `a = -A_ij` it solves `A x = b` by
belief propagation. Its messages are improper, with negative precision, and its rules take the
structured factorisation `q(out, in) q(a)`, which a constant `a` gives by itself.

```julia
node = factornode(GaussianCoupling, [(:out, x₂), (:in, x₁), (:a, constvar(-0.5))], ((:out, :in), (:a,)))
```

```@docs
GaussianCoupling
```
