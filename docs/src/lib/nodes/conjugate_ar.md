# [Conjugate Autoregressive node](@id lib-nodes-conjugate-ar)

The `ConjugateAR` node encodes the same **Bayesian autoregressive process** as the [Autoregressive node](@ref lib-nodes-ar):

```math
y_t \sim \mathcal{N}(\theta^\top x_t, \, \gamma^{-1})
```

The likelihood is identical to the [Autoregressive node](@ref lib-nodes-ar), but the AR coefficients `θ` and the transition precision `γ` enter through a **single** edge `w` that carries the joint `(θ, γ)` as an `MvNormalGamma` variable, instead of the two separate mean-field edges `θ` and `γ` of `AR`.

Keeping `(θ, γ)` joint makes the parameter update exactly conjugate: under the structured factorization `q(y, x) q(θ, γ)` the posterior `q(θ, γ)` is again `MvNormalGamma`. This is a Bayesian linear regression with unknown noise precision, where the regression statistics are the expected sufficient statistics of `q(y, x)`.

## [Interfaces](@id lib-nodes-conjugate-ar-interfaces)

| Interface | Alias | Role |
|-----------|-------|------|
| `y` | `out` | Current observation `yₜ` |
| `x` | — | Lagged state vector `(yₜ₋₁, …, yₜ₋ₚ)` |
| `w` | — | Joint parameters `(θ, γ)` distributed as `MvNormalGamma` |

## [Metadata](@id lib-nodes-conjugate-ar-meta)

`ConjugateAR` reuses `ARMeta` for the order and variate form, and — like `AR` — requires it to be passed explicitly:

```julia
y[t] ~ ConjugateAR(x[t], w) where { meta = ARMeta(Multivariate, order, ARsafe()) }
```

The state messages (`:y`, `:x`), the `(y, x)` joint marginal, and the average energy delegate to the existing `AR` rules through the effective `(q_θ, q_γ)` moments computed by [`ReactiveMP.conjugatear_effective_marginals`](@ref).

```@docs
ReactiveMP.ConjugateAR
ReactiveMP.conjugatear_effective_marginals
```
