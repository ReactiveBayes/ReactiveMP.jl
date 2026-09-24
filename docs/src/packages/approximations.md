# [Approximations](@id packages-approximations)

```@docs
MessagePassingRulesApproximations
```

`MessagePassingRulesApproximations` propagates means and covariances through a deterministic
function. It depends on no rule package and knows nothing of message passing: node packages,
such as the Delta node's, build their rules on it.

## Methods

```@docs
AbstractApproximationMethod
approximation_name
approximation_short_name
approximate
```

## The unscented transform

```@docs
Unscented
UT
UnscentedTransform
unscented_statistics
sigma_points_weights
```

## Local linearization

A first-order Taylor expansion of the function at the inputs' means, by automatic
differentiation. Where the unscented transform returns moments, linearization returns the linear
map, and the caller propagates the moments through it.

```@docs
Linearization
local_linearization
```

## Smoothing

```@docs
smoothRTS
```
