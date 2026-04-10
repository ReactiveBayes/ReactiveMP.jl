# [BinomialPolya node](@id lib-nodes-binomial-polya)

The `BinomialPolya` node implements a **Binomial likelihood with a logistic linear predictor**, augmented with a Pólya-Gamma auxiliary variable for tractable Bayesian inference:

```math
y \mid x, \beta, n \sim \mathrm{Binomial}\!\left(n,\; \sigma(x^\top \beta)\right)
```

where `σ` is the logistic (sigmoid) function, `x` is a feature vector, `β` is a weight vector with a Normal prior, and `n` is the number of trials.

## [Interfaces](@id lib-nodes-binomial-polya-interfaces)

| Interface | Role |
|-----------|------|
| `y` | Observed count (number of successes) |
| `x` | Feature vector |
| `n` | Number of trials |
| `β` | Weight vector (Normal prior) |

## [The Pólya-Gamma augmentation trick](@id lib-nodes-binomial-polya-augmentation)

Combining a Normal prior on `β` with a Binomial likelihood through a logistic link is not conjugate — the posterior has no closed form. The **Pólya-Gamma augmentation** (Polson et al., 2013) introduces a latent variable `ω ~ PG(n, x⊤β)` such that, conditional on `ω`, the likelihood becomes Gaussian. This makes the full-conditional update for `β` analytically tractable and allows the engine to perform exact conjugate message passing instead of sampling or variational approximation.

This is useful for:
- **Binomial regression** — modeling count data with a logistic link.
- **Binary classification** — as a special case with `n = 1`.

## [Meta and tuning](@id lib-nodes-binomial-polya-meta)

`BinomialPolyaMeta` controls the Monte Carlo estimation of the average energy:

| Field | Default | Effect |
|-------|---------|--------|
| `n_samples` | `1` | Number of samples for MC energy estimation. Increasing adds cost with diminishing accuracy benefit. |
| `rng` | `Random.default_rng()` | Random number generator. |

If no meta is provided (`meta = nothing`), the rules use posterior means instead of sampling, which yields very similar results at no extra cost.

```@docs
ReactiveMP.BinomialPolya
ReactiveMP.BinomialPolyaMeta
```
