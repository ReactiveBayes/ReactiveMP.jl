# The unscented transform

The unscented transform places `2d + 1` sigma points around the mean of a `d`-dimensional normal,
passes each through the function, and estimates the output's mean and covariance from weighted
sums. [`approximate`](@ref) returns the output's moments; [`unscented_statistics`](@ref) also
returns the cross-covariance between input and output, which [`smoothRTS`](@ref) uses.

```@docs
Unscented
UT
UnscentedTransform
approximate(::Unscented, ::F, ::Tuple, ::Tuple) where {F}
unscented_statistics
sigma_points_weights
```
