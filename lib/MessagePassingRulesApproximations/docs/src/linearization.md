# [Linearisation](@id approximations-linearisation)

Local linearisation replaces a function by its first-order Taylor expansion at the inputs'
means. A normal passes through a linear map exactly, so the approximation is in the function,
not in the moments.

## The method

Near the expansion point ``\hat{x}``, usually the mean,

```math
f(x) ≈ f(\hat{x}) + J (x - \hat{x}) = A x + b, \qquad A = J, \quad b = f(\hat{x}) - J \hat{x},
```

where ``J`` is the Jacobian of ``f`` at ``\hat{x}``: a derivative for a scalar function of a
scalar, a gradient row for a scalar function of a vector. A normal ``N(m, V)`` then maps to
``N(A m + b, A V A^\top)``. The Jacobian comes from automatic differentiation with ForwardDiff,
so `f` must accept its dual numbers, which any generic Julia function does.

[`approximate`](@ref)`(::Linearization, f, x̂)` returns the map `(A, b)`, and the caller
propagates the moments through it. For `exp` at `0`, the tangent line is `1 + x`:

```@example linearisation
using MessagePassingRulesApproximations

m, v = 0.0, 0.25
A, b = approximate(Linearization(), exp, (m,))
(A, b), (A * m + b, A * v * A')
```

The linearised mean is `exp(0) = 1`, while the exact mean is `exp(v / 2) ≈ 1.133`: the expansion
keeps no curvature. The error grows with the input's variance, and vanishes for an affine
function.

A function of several arguments is expanded in all of them at once. The columns of `A` follow
the arguments concatenated into one vector:

```@example linearisation
approximate(Linearization(), (x, y) -> x .- y, ([1.0, 2.0], [0.5, 0.5]))
```

## API

```@docs
Linearization
approximate(::Linearization, ::G, ::Tuple) where {G}
local_linearization
```
