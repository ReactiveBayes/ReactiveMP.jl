# MessagePassingRulesApproximations

```@docs
MessagePassingRulesApproximations
```

## Choosing a method

Each method answers one question: if `x` is normal with mean `m` and covariance `V`, what are
the mean and covariance of `f(x)`? For an affine `f` the answer is exact and normal. For any
other `f`, the distribution of `f(x)` is not normal, and each method approximates its moments
in a different way.

| method | needs | returns | suits |
|---|---|---|---|
| [`Unscented`](@ref) | `f` evaluated at `2d + 1` points | the output's mean and covariance | any function, no derivatives; accurate to second order |
| [`Linearization`](@ref) | `f` differentiable by ForwardDiff | the local linear map `(A, b)` | functions close to linear over the inputs' spread; exact for affine ones |
| [`GaussHermiteCubature`](@ref) | `f` evaluated at `p^d` points | weights and points, for any expectation | expectations and reweighted moments in a few dimensions |

The three methods on `f(x) = exp(x)`, for `x` normal with mean `0` and variance `0.25`. The
exact mean is `exp(m + v / 2)`:

```@example overview
using MessagePassingRulesApproximations

m, v = 0.0, 0.25
exact = exp(m + v / 2)

unscented = first(approximate(Unscented(), exp, (m,), (v,)))

A, b = approximate(Linearization(), exp, (m,))
linearised = A * m + b

gh = ghcubature(20)
cubature = sum(w * exp(x) for (w, x) in zip(getweights(gh, m, v), getpoints(gh, m, v)))

(; exact, unscented, linearised, cubature)
```

Linearisation keeps only the slope of `exp` at the mean, so it misses the curvature that raises
the mean. The unscented transform captures the curvature to second order. Twenty cubature points
reproduce the exact value to machine precision, since `exp` is well approximated by a
polynomial over the normal's spread.

A node package pairs the first two with [`smoothRTS`](@ref) to correct an input's marginal from a
backward message, and uses the third for expectations its rules cannot take in closed form.

## The common interface

Every method is an [`AbstractApproximationMethod`](@ref), with a name for display.

```@example overview
approximation_name(Unscented()), approximation_short_name(ghcubature(5))
```

```@docs
AbstractApproximationMethod
approximation_name
approximation_short_name
```
