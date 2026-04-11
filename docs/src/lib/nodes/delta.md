# [Delta node](@id lib-nodes-delta)

The delta node encodes a **deterministic functional relationship** between variables. Where a stochastic node represents `p(y | x)`, a delta node asserts that `y = f(x₁, …, xₙ)` exactly. Any Julia function `f` can be used.

```julia
z ~ f(x, y)   # z is deterministically f(x, y)
```

Because `f` is not a probability distribution, the standard closed-form message computation does not apply. The engine must **approximate** the outgoing messages. The approximation method is specified via [`DeltaMeta`](@ref):

```julia
z ~ f(x, y) where { meta = DeltaMeta(method = Linearization()) }
z ~ f(x, y) where { meta = DeltaMeta(method = Unscented()) }
z ~ f(x, y) where { meta = DeltaMeta(method = CVI(...)) }
```

## [Choosing an approximation method](@id lib-nodes-delta-methods)

| Method | Best for | What it needs |
|--------|----------|---------------|
| [`Linearization`](@ref) | Smooth `f` that is approximately linear near the operating point | Jacobian, computed via ForwardDiff automatically |
| [`Unscented`](@ref) / [`UT`](@ref) | Nonlinear but smooth `f` in moderate dimension | Sigma points; no derivatives required |
| [`CVI`](@ref) | Black-box or non-differentiable `f`, high dimension | Stochastic gradient estimator; requires an optimizer |
| [`CVIProjection`](@ref) | Same as `CVI` with the result projected onto an exponential family member | Same as `CVI` |
| [`LaplaceApproximation`](@ref) | Unimodal posteriors; `f` differentiable | Second-order Taylor expansion at the mode |

When `f` has a known analytical inverse `f⁻¹`, you can pass it as the `inverse` keyword to skip the backward approximation entirely:

```julia
z ~ f(x) where { meta = DeltaMeta(method = Linearization(), inverse = f_inv) }
```

Without an inverse, the backward (input) messages are computed via the [RTS smoother](@ref ReactiveMP.smoothRTS) (Petersen et al., 2018).

## [Multi-input delta nodes](@id lib-nodes-delta-manyof)

When a delta node has more than one input, the `@rule` macro receives the inputs bundled in a [`ReactiveMP.ManyOf`](@ref) container. This lets the rule dispatch on the collection of input messages rather than individually:

```julia
@rule DeltaFn{typeof(f)}(:out, Marginalisation) (
    m_ins::ReactiveMP.ManyOf,
    meta::DeltaMeta{<:Linearization},
) = begin
    # m_ins[1], m_ins[2], ... are the individual input messages
    ...
end
```

See the [Message update rules](@ref lib-rules) page for how to define rules with `@rule`.

!!! note
    The delta node is [`Deterministic`](@ref) and does not contribute to the Bethe free energy directly. It only transforms information between variables.

For the full API of approximation methods (CVI, Unscented, Linearization, etc.), see [Approximation methods](@ref lib-approximations).

```@docs
ReactiveMP.DeltaMeta
ReactiveMP.ManyOf
ReactiveMP.smoothRTS
ReactiveMP.CVIApproximationDeltaFnRuleLayout
ReactiveMP.log_approximate
ReactiveMP.DeltaFnDefaultRuleLayout
ReactiveMP.DeltaFnDefaultKnownInverseRuleLayout
```

## [SoftDot — soft dot product node](@id lib-nodes-delta-softdot)

`SoftDot` is a **stochastic** counterpart of the deterministic dot-product delta node.
Instead of enforcing `y = θ ⋅ x` exactly, it relaxes the constraint by adding
Gaussian noise:

```math
y \sim \mathcal{N}(\theta^\top x,\; \gamma^{-1})
```

where `γ` is a precision parameter. This relaxation makes message computation
tractable in closed form for both belief propagation and variational message passing.

`SoftDot` is the natural building block for Bayesian linear regression, the
[Autoregressive node](@ref lib-nodes-ar), and other models that need a
differentiable but noisy dot product.

Interfaces: `y` (result), `θ` (alias `theta`), `x`, `γ` (alias `gamma`).

```@docs
SoftDot
softdot
```
