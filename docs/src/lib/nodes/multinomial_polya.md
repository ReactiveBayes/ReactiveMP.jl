# [MultinomialPolya node](@id lib-nodes-multinomial-polya)

The `MultinomialPolya` node implements a **Multinomial likelihood with a softmax linear predictor**, augmented with Pólya-Gamma auxiliary variables for tractable Bayesian inference:

```math
x \mid N, \psi \sim \mathrm{Multinomial}\!\left(N,\; \mathrm{softmax}(\psi)\right)
```

where `x` is a count vector, `N` is the total number of trials, and `ψ` is a latent vector with a Normal prior.

## [Interfaces](@id lib-nodes-multinomial-polya-interfaces)

| Interface | Role |
|-----------|------|
| `x` | Observed count vector |
| `N` | Total number of trials |
| `ψ` | Softmax weight vector (Normal prior) |

## [The Pólya-Gamma augmentation trick](@id lib-nodes-multinomial-polya-augmentation)

A Normal prior on `ψ` combined with a Multinomial likelihood through a softmax link is not conjugate. The **Pólya-Gamma augmentation** (Polson et al., 2013) uses a set of latent Pólya-Gamma variables — one per category — such that, conditional on these variables, the likelihood factorizes into a product of Gaussian terms. This restores conjugacy with the Normal prior on `ψ` and allows closed-form VMP updates.

The key advantage over Monte Carlo methods is that inference remains deterministic and converges smoothly, making it suitable for models where `ψ` is a latent variable that must be marginalized.

Typical use cases:
- **Multinomial regression** — predicting category counts from a feature vector.
- **Topic models** — where category probabilities are the softmax of a Gaussian-distributed topic vector.

## [Meta and tuning](@id lib-nodes-multinomial-polya-meta)

`MultinomialPolyaMeta` controls the number of cubature points used to integrate out the Pólya-Gamma variables:

| Field | Default | Effect |
|-------|---------|--------|
| `ncubaturepoints` | `21` ([`MULTINOMIAL_POLYA_CUBATURE_POINTS`](@ref ReactiveMP.MULTINOMIAL_POLYA_CUBATURE_POINTS)) | More points → higher accuracy, higher cost. Reduce to `7` or `9` for faster but less accurate inference. |

The default of 21 cubature points balances accuracy and speed for typical problem sizes.

```@docs
ReactiveMP.MultinomialPolya
ReactiveMP.MultinomialPolyaMeta
ReactiveMP.MULTINOMIAL_POLYA_CUBATURE_POINTS
```
