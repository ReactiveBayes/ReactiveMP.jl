# [Approximation methods](@id lib-approximations)

Approximation methods are used when exact message computation is intractable — most commonly inside [Delta nodes](@ref lib-nodes-delta), where the factor is a nonlinear or non-conjugate function `y = f(x)`. Each method trades off accuracy, computational cost, and assumptions about `f`.

All approximation methods are passed through [`DeltaMeta`](@ref) or [`FlowMeta`](@ref).

## [Choosing a method](@id lib-approximations-choosing)

| Method | Best for | Dimensionality | Requires |
|--------|----------|----------------|---------|
| `Linearization` | Smooth, nearly linear `f` | Any | ForwardDiff (auto) |
| `Unscented` | Smooth nonlinear `f` | Low–moderate | Nothing |
| `GaussHermiteCubature` | Univariate integrals with Gaussian inputs | Univariate | Point count `p` |
| `GaussLaguerreQuadrature` | Integrals over `[0, ∞)` | Univariate | Point count `n` |
| `srcubature` / `SphericalRadialCubature` | Multivariate Gaussian integrals | Multivariate | Nothing |
| `LaplaceApproximation` | Unimodal posteriors, differentiable `f` | Any | ForwardDiff + Optim |
| [`CVI`](@ref) | Black-box or non-differentiable `f` | Any | Optimizer + gradient |
| [`CVIProjection`](@ref) | CVI + exponential family projection | Any | ExponentialFamilyProjection.jl |
| [`ImportanceSamplingApproximation`](@ref) | General expectations via sampling | Any | Proposal distribution |

## [Deterministic approximations](@id lib-approximations-deterministic)

### Linearization

[`Linearization`](@ref) approximates `f` by its first-order Taylor expansion around the current operating point. Jacobians are computed automatically using [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl). This is the default method for [`FlowMeta`](@ref) and a common choice for [`DeltaMeta`](@ref) when `f` is smooth and not highly nonlinear.

### Unscented transform

[`Unscented`](@ref) (also [`UT`](@ref) / [`UnscentedTransform`](@ref)) propagates a deterministic set of *sigma points* through `f` and fits a Gaussian to the outputs. It captures mean and covariance through nonlinearities more accurately than linearization, without requiring derivatives. The number of sigma points scales linearly with the input dimension.

### Gauss-Hermite cubature

`GaussHermiteCubature` computes expectations of the form `∫ g(x) N(x; μ, σ²) dx` using a fixed set of quadrature points and weights optimized for Gaussian measures. It is exact for polynomials up to a certain degree determined by the number of points `p`:

```julia
DeltaMeta(method = GaussHermiteCubature(21))  # 21-point rule
```

### Gauss-Laguerre quadrature

`GaussLaguerreQuadrature` computes expectations over the half-line `[0, ∞)` — useful when the input has a Gamma distribution or similar semi-infinite support.

### Spherical radial cubature

`srcubature()` constructs a spherical-radial cubature rule for multivariate Gaussian integrals, using `2d + 1` deterministic points (where `d` is the input dimension). It provides a good balance between accuracy and cost for moderate dimensions.

### Laplace approximation

`LaplaceApproximation` finds the mode of the log-unnormalized posterior (using [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl)) and fits a Gaussian at that mode using the local curvature (via ForwardDiff). Best for unimodal, differentiable posteriors.

## [Stochastic approximations](@id lib-approximations-stochastic)

### CVI — Constrained Variational Inference

[`CVI`](@ref) and [`ProdCVI`](@ref) approximate messages using stochastic gradient optimization of a variational objective. A gradient estimator (default: [`ForwardDiffGrad`](@ref)) computes the gradient of the log-likelihood with respect to the natural parameters of the approximating distribution. An optimizer (default: `Adam`) applies gradient steps until convergence.

```julia
DeltaMeta(method = CVI(
    rng,          # random number generator
    n_samples,    # samples for gradient estimation
    n_iterations, # gradient steps per message update
    Adam(params), # optimizer
))
```

!!! note
    Loading [Optimisers.jl](https://github.com/FluxML/Optimisers.jl) unlocks additional optimizers for use with CVI. See the [Extensions](@ref extra-extensions) page.

### CVIProjection

[`CVIProjection`](@ref) extends CVI by projecting the result onto the nearest member of a chosen exponential family, using [ExponentialFamilyProjection.jl](https://github.com/ReactiveBayes/ExponentialFamilyProjection.jl). This guarantees that the output is a valid member of the target family.

!!! note
    `CVIProjection` requires the `ExponentialFamilyProjection` package to be loaded. Without it, using `CVIProjection` in a delta node will throw an informative error.

### Importance sampling

[`ImportanceSamplingApproximation`](@ref) estimates expectations by drawing samples from a proposal distribution and reweighting. It is the most flexible method but converges slowly in high dimensions.

## [API reference](@id lib-approximations-api)

```@docs
ReactiveMP.Linearization
ReactiveMP.local_linearization
ReactiveMP.Unscented
ReactiveMP.sigma_points_weights
ReactiveMP.UT
ReactiveMP.UnscentedTransform
ReactiveMP.CVI
ReactiveMP.ProdCVI
ReactiveMP.ForwardDiffGrad
ReactiveMP.CVIProjection
ReactiveMP.CVISamplingStrategy
ReactiveMP.FullSampling
ReactiveMP.MeanBased
ReactiveMP.ProposalDistributionContainer
ReactiveMP.cvi_setup!
ReactiveMP.cvi_update!
```
