# [Autoregressive node](@id lib-nodes-ar)

The `AR` node (also exported as `Autoregressive`) encodes a **Bayesian autoregressive process** of order `p`:

```math
y_t \sim \mathcal{N}(\theta^\top x_t, \, \gamma^{-1})
```

where `yₜ` is the current observation, `xₜ = (yₜ₋₁, …, yₜ₋ₚ)` is the vector of `p` lagged values, `θ` is the vector of AR coefficients, and `γ` is the observation precision.

This node is the natural building block for **time series models** such as AR(p), latent AR processes, and state-space models with autoregressive dynamics.

## [Interfaces](@id lib-nodes-ar-interfaces)

| Interface | Alias | Role |
|-----------|-------|------|
| `y` | `out` | Current observation `yₜ` |
| `x` | — | Lagged state vector `(yₜ₋₁, …, yₜ₋ₚ)` |
| `θ` | — | AR coefficient vector (length `p`) |
| `γ` | — | Observation precision (scalar) |

## [Metadata](@id lib-nodes-ar-meta)

`ARMeta` is required and must be passed explicitly — the node has no default meta:

```julia
y[t] ~ AR(x[t], θ, γ) where { meta = ARMeta(Multivariate, order, ARsafe()) }
```

The constructor takes:
- `Univariate` or `Multivariate` — variate form (determines how `x` and `y` are interpreted).
- `order` — the AR order `p` (must equal 1 for `Univariate`).
- `ARsafe()` or `ARunsafe()` — numerical stability mode (`ARsafe` adds a small regularization to avoid singular matrices; `ARunsafe` is faster but may be numerically fragile).

## [Univariate vs multivariate](@id lib-nodes-ar-variate)

`ARMeta{Univariate}` treats `y` and the first element of `x` as scalars, with order forced to 1. This is an AR(1) model.

`ARMeta{Multivariate}` uses the full companion-matrix representation to handle AR(p) for `p > 1`. The state vector `x` has length `p`, and the AR process is embedded as a linear state-space model. See [`CompanionMatrix`](@ref) for the underlying algebraic structure.

## [State vector slicing](@id lib-nodes-ar-slicing)

The [`ReactiveMP.ar_unit`](@ref) and [`ReactiveMP.ar_slice`](@ref) utilities extract specific parts of the joint state vector in the multivariate setting:

- [`ReactiveMP.ar_unit`](@ref) — returns an appropriately shaped zero vector or matrix for initializing accumulators.
- [`ReactiveMP.ar_slice`](@ref) — extracts a subvector or submatrix from a joint mean/covariance. This is used inside rules to separate the `y` part from the `x` part of the joint Gaussian `q(y, x)`.

These are internal helpers that surface when writing custom rules for AR-based models.

```@docs
ReactiveMP.ar_unit
ReactiveMP.ar_slice
```
