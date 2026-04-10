# [Extensions and ecosystem integration](@id extra-extensions)

`ReactiveMP.jl` activates extra functionality when other Julia packages are loaded alongside it. These are implemented as Julia [package extensions](https://pkgdocs.julialang.org/v1/creating-packages/#Conditional-dependencies) (weak dependencies) and require no additional configuration — simply `using` the relevant package is enough.

## [Optimisers.jl](@id extra-extensions-optimisers)

**Package:** [`Optimisers.jl`](https://github.com/FluxML/Optimisers.jl)

**What it provides:** Gradient-based optimizers (Adam, ADAM, NADAM, RMSProp, etc.) compatible with the [`CVI`](@ref) approximation method.

```julia
using ReactiveMP, Optimisers

meta = DeltaMeta(method = CVI(
    rng,
    n_samples    = 100,
    n_iterations = 50,
    opt          = Optimisers.Adam(0.01),   # ← any Optimisers.jl rule
))

y ~ f(x) where { meta = meta }
```

**How it works internally:** The extension implements `ReactiveMP.cvi_setup` and `ReactiveMP.cvi_update!` for `Optimisers.AbstractRule`, delegating to `Optimisers.init` and `Optimisers.apply!`. This maps the Optimisers.jl stateful optimizer API onto the CVI update loop.

## [DiffResults.jl](@id extra-extensions-diffresults)

**Package:** [`DiffResults.jl`](https://github.com/JuliaDiff/DiffResults.jl) — loaded automatically when `ForwardDiff.jl` is present.

**What it provides:** Faster derivative computation for the [`ForwardDiffGrad`](@ref) gradient estimator inside CVI, in the special case where all inputs are Gaussian distributions.

When `DiffResults` is available, `ForwardDiffGrad` uses `DiffResults.DiffResult` as an output buffer for in-place differentiation, avoiding redundant forward passes. This can meaningfully reduce the per-iteration cost of CVI in purely Gaussian models.

No explicit configuration is needed — the extension activates automatically whenever `ForwardDiff` (and transitively `DiffResults`) is loaded into the session.

## [ExponentialFamilyProjection.jl](@id extra-extensions-projection)

**Package:** [`ExponentialFamilyProjection.jl`](https://github.com/ReactiveBayes/ExponentialFamilyProjection.jl)

**What it provides:** Enables [`CVIProjection`](@ref) for use inside [Delta nodes](@ref lib-nodes-delta).

[`CVIProjection`](@ref) extends CVI by projecting the resulting approximate message onto the nearest member of a target exponential family. This projection step requires `ExponentialFamilyProjection.jl`. Without it, placing `CVIProjection` in a delta node raises an informative error.

```julia
using ReactiveMP, ExponentialFamilyProjection

meta = DeltaMeta(method = CVIProjection(
    sampling_strategy = FullSampling(100),
))

y ~ f(x) where { meta = meta }
```

The extension also defines `prod` rules on `DivisionOf` objects needed for the backward message computation in the delta node, and registers `CVIProjection` as compatible with the delta node approximation framework.
