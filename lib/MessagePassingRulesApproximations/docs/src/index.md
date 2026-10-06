# MessagePassingRulesApproximations

A rule for a nonlinear node needs the
[pushforward](@extref MessagePassingRulesBase glossary-pushforward) of a normal through a
function: if `x` is normal, what are the mean and covariance of `f(x)`? This package computes it,
and expectations under a normal, with three methods. It works on means and covariances, plain
numbers and arrays, and knows nothing of message passing. Node packages, such as the Delta
node's, build their rules on it.

This site is for authors of such rules, and for anyone who wants to compare the methods on a
function of their own.

```@docs
MessagePassingRulesApproximations
```

## A first approximation

The three methods on `f(x) = exp(x)`, for `x` normal with mean `0` and variance `0.25`. The exact
mean is `exp(m + v / 2)`:

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
reproduce the exact value to machine precision, since `exp` is close to a polynomial over the
normal's spread.

## The site

- [Choosing a method](@ref): what each method needs and costs, and the three compared on one
  function against its exact moments.
- [The unscented transform](@ref): sigma points, their weights, and moments through a function.
- [Linearisation](@ref approximations-linearisation): the first-order expansion of a function,
  by automatic differentiation.
- [Gauss–Hermite cubature](@ref): expectations under a normal, and the moments of a normal
  reweighted by a function.
- [Smoothing](@ref): the Rauch–Tung–Striebel correction of an input's belief from a backward
  message, with either transform.

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
