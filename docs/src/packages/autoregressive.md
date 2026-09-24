# [Autoregressive](@id packages-autoregressive)

```@docs
AutoregressiveMessagePassingRules
```

`AR` is a Bayesian autoregressive process, `yₜ ~ N(θᵀxₜ, γ⁻¹)`, with the coefficients `θ` and
the precision `γ` on edges of their own; `ConjugateAR` keeps `(θ, γ)` joint on one
`MvNormalGamma` edge `w`. Both run under `ARVMP`, which the model must give, since the variate
form and the order are properties of the model rather than of the node. v6 called it `ARMeta`.

```@docs
AR
ConjugateAR
ARVMP
ARsafe
ARunsafe
```

!!! note
    v6's `ARunsafe` joint `q(y, x)` was wrong: its block inversion had a Schur complement of
    the wrong sign, so it disagreed with `ARsafe` even for a univariate AR(1). It is computed by
    the Kalman gain now, exact, and agrees with `ARsafe` up to `ARsafe`'s regularisation.
