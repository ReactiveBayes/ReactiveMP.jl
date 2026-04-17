# [BIFM node](@id lib-nodes-bifm)

The **Backward Information Forward Marginals (BIFM)** node implements an efficient Kalman smoothing step for linear Gaussian state-space models. It fuses all factor contributions within a single time slice — observation likelihood, state transition, and the backward information from future time steps — into one node, enabling correct smoothed marginals without a separate backward pass.

## [Model structure](@id lib-nodes-bifm-model)

The BIFM node has four interfaces:

| Interface | Role |
|-----------|------|
| `out` | Latent output (observation) of the time slice |
| `in` | Latent input to the time slice (e.g., a control signal) |
| `zprev` | Previous latent state `zₜ₋₁` |
| `znext` | Next latent state `zₜ` (carries backward information from future) |

The state-space equations encoded by the node are:

```math
z_t = A \, z_{t-1} + B \, u_t, \qquad x_t = C \, z_t
```

where `A`, `B`, and `C` are the transition, input, and output matrices stored in [`BIFMMeta`](@ref).

## [Usage](@id lib-nodes-bifm-usage)

The BIFM node must be used together with [`BIFMHelper`](@ref), which carries backward smoothing information between time steps. A typical model looks like:

```julia
z_prior ~ MvNormalMeanPrecision(zeros(latent_dim), diagm(ones(latent_dim)))
z_tmp   ~ BIFMHelper(z_prior)
z_prev  = z_tmp

for i in 1:nr_samples
    u[i]  ~ MvNormalMeanPrecision(μu, Wu)
    xt[i] ~ BIFM(u[i], z_prev, z[i]) where { meta = BIFMMeta(A, B, C) }
    x[i]  ~ MvNormalMeanPrecision(xt[i], Wx)
    z_prev = z[i]
end
```

!!! note
    When subscribing to marginals, subscribe in the order `z`, `out`, `in` before subscribing to the free energy score function. This ordering ensures that the backward information is propagated correctly before the score is evaluated.

## [Relationship to ContinuousTransition](@id lib-nodes-bifm-vs-ctransition)

The [`ContinuousTransition`](@ref) node encodes a single linear-Gaussian transition `y ~ N(K(a)·x, W⁻¹)` where the transition matrix can itself be a latent variable. BIFM is a more specialized node: the matrices `A`, `B`, `C` are fixed (passed through meta), but the node efficiently handles the full time-slice factor, including the smoothing backward pass. Use `ContinuousTransition` when the transition matrix is uncertain and must be inferred; use BIFM when the structure is known and smoothing efficiency matters.

!!! note
    See also the [BIFM tutorial](https://reactivebayes.github.io/RxInfer.jl/stable/examples/overview/) in the RxInfer.jl documentation for a comprehensive guide.

```@docs
ReactiveMP.BIFM
ReactiveMP.BIFMMeta
ReactiveMP.BIFMHelper
```
