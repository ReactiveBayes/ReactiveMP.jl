# The unscented transform

The unscented transform estimates the mean and covariance of `f(x)` for a normal `x` without
differentiating `f`. It places `2d + 1` *sigma points* around the mean of a `d`-dimensional
normal `N(m, V)`, passes each through `f`, and forms weighted sums of the results.

## The method

With a scaling ``λ = α^2 (d + κ) - d`` and ``L`` a square root of ``(d + λ) V``, so that
``L L^\top = (d + λ) V``, the sigma points are the mean and a pair along each column ``L_i``:

```math
χ_0 = m, \qquad χ_i = m + L_i, \qquad χ_{d + i} = m - L_i, \qquad i = 1, \ldots, d.
```

The output's moments are weighted sums over the transformed points ``y_i = f(χ_i)``:

```math
\tilde{m} = \sum_i W^{(m)}_i y_i, \qquad
\tilde{V} = \sum_i W^{(c)}_i (y_i - \tilde{m})(y_i - \tilde{m})^\top,
```

with ``W^{(m)}_0 = λ / (d + λ)``, ``W^{(c)}_0 = W^{(m)}_0 + 1 - α^2 + β`` and
``W^{(m)}_i = W^{(c)}_i = 1 / (2(d + λ))`` for the other points. The keywords `alpha`, `beta`
and `kappa` of [`Unscented`](@ref) are ``α``, ``β`` and ``κ``. The estimates are exact when `f`
is affine, and match the true moments to second order in general.

[`sigma_points_weights`](@ref) returns the points and both sets of weights. For a scalar normal
with mean `1` and variance `0.5`, and `alpha = 1`, the three points lie at the mean and one
standard deviation times ``\sqrt{1 + λ}`` on either side:

```@example unscented
using MessagePassingRulesApproximations

points, weights_mean, weights_cov = sigma_points_weights(Unscented(alpha = 1.0), 1.0, 0.5)
```

## Moments through a function

[`approximate`](@ref) returns the output's mean and covariance. The classic example converts a
position from polar to Cartesian coordinates, with a radius known to within `0.1` and an angle
known to within about `0.55` radians:

```@example unscented
polar_to_cartesian(x) = [x[1] * cos(x[2]), x[1] * sin(x[2])]

m, V = [1.0, 0.0], [0.01 0.0; 0.0 0.3]
approximate(Unscented(), polar_to_cartesian, (m,), (V,))
```

The exact mean of the first coordinate is ``\mathrm{E}[r] \, \mathrm{E}[\cos θ] = \exp(-0.3 / 2)
≈ 0.861``. The unscented estimate, `0.85`, follows the curvature of the circle towards the origin;
a linearisation at the mean gives `1.0` ([Linearisation](@ref approximations-linearisation)).

Several inputs are passed as several entries of the tuples, and are treated as one normal whose
covariance is block diagonal:

```@example unscented
approximate(Unscented(), (x, y) -> x * y, (1.0, 2.0), (0.1, 0.2))
```

[`unscented_statistics`](@ref) also returns the cross-covariance between the inputs and the
output, which [`smoothRTS`](@ref) needs:

```@example unscented
unscented_statistics(Unscented(), x -> 2x + 1, (1.0,), (0.5,))
```

## API

```@docs
Unscented
UT
UnscentedTransform
approximate(::Unscented, ::F, ::Tuple, ::Tuple) where {F}
unscented_statistics
sigma_points_weights
MessagePassingRulesApproximations.sigma_point_parameters
```
