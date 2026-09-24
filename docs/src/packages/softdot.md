# [SoftDot](@id packages-softdot)

```@docs
SoftDotMessagePassingRules
```

`SoftDot` is `y ~ N(θ ⋅ x, 1/γ)`: a dot product softened by Gaussian noise, which gives closed-form
variational messages where a deterministic dot product would need an approximation. Its rules
are variational, under mean-field or the structured `q(y, x)`; it has no belief-propagation rules.
It is also available as `softdot`.

```@docs
SoftDot
softdot
```
