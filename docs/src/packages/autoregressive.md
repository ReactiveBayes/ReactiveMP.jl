# [Autoregressive](@id packages-autoregressive)

```@docs
AutoregressiveMessagePassingRules
```

`AR` is a Bayesian autoregressive process, `yₜ ~ N(θᵀxₜ, γ⁻¹)`, with the coefficients `θ` and
the precision `γ` on edges of their own; `ConjugateAR` keeps `(θ, γ)` joint on one
`MvNormalGamma` edge `w`. Both run under `ARVMP`, which the model must give, since the variate
form and the order are properties of the model rather than of the node.

```@docs
AR
ConjugateAR
ARVMP
ARsafe
ARunsafe
```

`ARunsafe`'s joint `q(y, x)` is computed through the Kalman gain, exactly, and agrees with
`ARsafe` up to `ARsafe`'s regularisation.
