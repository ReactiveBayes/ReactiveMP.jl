
# [Variables](@id lib-variables)

Variables are fundamental building blocks of a factor graph. Each variable represents either a latent quantity to be inferred, an observed data point, or a fixed constant. All variable types are subtypes of [`ReactiveMP.AbstractVariable`](@ref).

```@docs
ReactiveMP.AbstractVariable
```

## [Random variables](@id lib-variables-random)

Random variables represent latent (unobserved) quantities in the model. During inference, messages flow through them to update the marginal belief.

```@docs
ReactiveMP.RandomVariable
ReactiveMP.randomvar
```

## [Data variables](@id lib-variables-data)

Data variables represent observed quantities. Their value is not fixed at creation time and can be updated later via [`update!`](@ref).

```@docs
ReactiveMP.DataVariable
ReactiveMP.datavar
ReactiveMP.update!
```

## [Constant variables](@id lib-variables-constant)

Constant variables hold a fixed value, wrapped in a `PointMass` distribution. Messages from constant variables are always marked as clamped.

```@docs
ReactiveMP.ConstVariable
ReactiveMP.constvar
```
