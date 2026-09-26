# MessagePassingRulesApproximations

```@docs
MessagePassingRulesApproximations
```

## Choosing a method

All three methods approximate a function of normal inputs; they differ in what they need of the
function and in what they return.

| method | needs | returns | suits |
|---|---|---|---|
| [`Unscented`](@ref) | `f` evaluated at `2d + 1` points | the output's mean and covariance | any function, no derivatives; accurate to second order |
| [`Linearization`](@ref) | `f` differentiable by ForwardDiff | the local linear map `(A, b)` | functions close to linear over the inputs' spread; exact for affine ones |
| [`GaussHermiteCubature`](@ref) | `f` evaluated at `p^d` points | weights and points, for any expectation | expectations and reweighted moments in a few dimensions |

A node package pairs the first two with [`smoothRTS`](@ref) to correct an input's marginal from a
backward message, and uses the third for expectations its rules cannot take in closed form.

## The common interface

Every method is an [`AbstractApproximationMethod`](@ref), with a name for display.

```@docs
AbstractApproximationMethod
approximation_name
approximation_short_name
```
