# [Continuous transition node](@id lib-nodes-ctransition)

The `ContinuousTransition` node encodes a **linear (or nonlinear) Gaussian state transition**:

```math
y \sim \mathcal{N}(K(a) \cdot x, \, W^{-1})
```

It transforms an `m`-dimensional input vector `x` into an `n`-dimensional output vector `y` via a learned matrix `K(a)`, where `a` is a latent vector and `K` is a user-supplied transformation function. The precision matrix `W` controls the amount of jitter in the transition.

This node is the continuous-state counterpart of [`DiscreteTransition`](@ref) and the primary building block for **Kalman-filter-style state-space models** where the transition matrix is uncertain and must be inferred.

## [Interfaces](@id lib-nodes-ctransition-interfaces)

| Interface | Role |
|-----------|------|
| `y` | `n`-dimensional output state |
| `x` | `m`-dimensional input state |
| `a` | Vector parameterizing the transition matrix via `K(a)` |
| `W` | `n×n` precision matrix of the transition noise |

## [Specifying the transformation](@id lib-nodes-ctransition-transformation)

The transformation `K(a)` is passed through [`ContinuousTransitionMeta`](@ref) (alias: `CTMeta`). The function must return an `n×m` matrix. For example:

```julia
# Unstructured: reshape a length-4 vector into a 2×2 matrix
transformation = a -> reshape(a, 2, 2)

a ~ MvNormalMeanCovariance(zeros(4), Diagonal(ones(4)))
y ~ ContinuousTransition(x, a, W) where { meta = CTMeta(transformation) }
```

When the matrix has known structure, `K(a)` can encode it explicitly:

```julia
# Rotation matrix parameterized by a single angle
transformation = a -> [cos(a[1]) -sin(a[1]); sin(a[1]) cos(a[1])]

a ~ MvNormalMeanCovariance([0.0], [1.0;;])
y ~ ContinuousTransition(x, a, W) where { meta = CTMeta(transformation) }
```

!!! note
    Even for scalar transitions, `a` must be a vector (length 1). Use `MvNormal` rather than `Normal` for the prior on `a`.

## [Factorization constraints](@id lib-nodes-ctransition-factorization)

The node supports two factorization assumptions:

**Mean-field** — all variables are treated as independent:
```julia
q(y, x, a, W) = q(y)q(x)q(a)q(W)
```

**Structured** — the joint `q(y, x)` is kept intact (useful for Kalman smoothing):
```julia
q(y, x, a, W) = q(y, x)q(a)q(W)
```

## [Companion matrix](@id lib-nodes-ctransition-companion)

For autoregressive-style transitions, the companion matrix representation converts an AR coefficient vector into a state transition matrix. See [`CompanionMatrix`](@ref) in the algebra utilities and the [Autoregressive node](@ref lib-nodes-ar) for a specific application.

```@docs
ReactiveMP.ContinuousTransition
ReactiveMP.ContinuousTransitionMeta
ReactiveMP.ctcompanion_matrix
```
