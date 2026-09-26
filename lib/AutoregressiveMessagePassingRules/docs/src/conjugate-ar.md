```@meta
DocTestSetup = :(using AutoregressiveMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily)
```

# [ConjugateAR](@id conjugate-ar-node)

## Overview

[`ConjugateAR`](@ref) is the likelihood of [`AR`](@ref) with the coefficients `θ` and the noise
precision `γ` joint on one edge `w = (θ, γ)`. With a normal-gamma prior on `w` the parameter
update is conjugate: the posterior `q(w)` is the normal-gamma posterior of a Bayesian linear
regression with unknown noise precision, with the states' expected sufficient statistics in
place of the data. Use it where `θ` and `γ` should stay coupled, rather than factorised as
`q(θ) q(γ)`.

## Definition

For the state `xₜ = (yₜ₋₁, …, yₜ₋ₚ)` and `w = (θ, γ)`:

```math
p(y \mid x, w) = \mathcal{N}\big(y_1 \mid \theta^\top x, \, \gamma^{-1}\big),
```

with `y = A(θ) x + ε` in companion form, noise on the first component only, as for
[`AR`](@ref). Averaged over `q(y, x)`, the node is a normal-gamma factor in `w` with statistics

```math
C = \langle x x^\top \rangle, \qquad b = \langle x\, y_1 \rangle, \qquad a = \langle y_1^2 \rangle,
```

that is `Λ = C`, `μ = C⁻¹ b`, shape `(3 - p)/2` and rate `(a - bᵀ C⁻¹ b)/2`.

## Interfaces

| name | aliases | meaning | messages and marginals its rules take |
|---|---|---|---|
| `y` | `out` | the current state | a multivariate normal message; `q(y)` with a mean and covariance |
| `x` | | the previous state | a multivariate normal message; `q(x)` with a mean and covariance |
| `w` | | the coefficients and the precision | `q(w)`, an `MvNormalGamma` of dimension `p` |

The joint marginal `q(y, x)` is a multivariate normal of dimension `2p`, `y` first.

## Algorithm

The node shares [`AR`](@ref)'s algorithm, [`ARVMP`](@ref)`(form, order, stype)`, which the
model must name, with every argument required. `form` must be `Multivariate`, an AR(1)
included, since `q(w)` has a vector `θ`; `stype` is [`ARsafe`](@ref)`()` or
[`ARunsafe`](@ref)`()`, as for [`AR`](@ref).

## Supported rules

```@example
using MessagePassingRulesBase, AutoregressiveMessagePassingRules # hide
MessagePassingRulesBase.rule_coverage(ConjugateAR)
```

Every rule is under [`ARVMP`](@ref). The rules towards `y` and `x` exist for the structured
factorisation `q(y, x) q(w)`, as belief propagation between them, and for the mean field; they
and the joint marginal `q(y, x)` are [`AR`](@ref)'s, computed from the marginals of `θ` and `γ`
that `q(w)` implies. The rule towards `w` and the average energy need the structured
factorisation `q(y, x) q(w)`.

## Example

The message towards `w` of an AR(1) from the joint marginal of the states:

```jldoctest
julia> result = @call_message_update_rule(
           node = ConjugateAR, target = :w, algorithm = ARVMP(Multivariate, 1, ARsafe()),
           clusters = ((:y, :x) => MvNormalMeanCovariance([1.0, 0.5], [1.0 0.0; 0.0 1.0]),),
       );

julia> μ, Λ, α, β = params(getresult(result));

julia> μ ≈ [0.4] && Λ ≈ fill(1.25, 1, 1) && α ≈ 1.0 && β ≈ 0.9
true
```

In a model, with RxInfer, `w` takes a normal-gamma prior and the states stay joint:

```julia
using RxInfer, AutoregressiveMessagePassingRules

@model function latent_conjugate_ar(y, order, γ)
    c = zeros(order); c[1] = 1.0
    w ~ MvNormalGamma(zeros(order), diageye(order), 2.0, 1.0)
    x0 ~ MvNormal(mean = zeros(order), precision = diageye(order))
    x_prev = x0
    for i in eachindex(y)
        x[i] ~ ConjugateAR(x_prev, w)
        y[i] ~ Normal(mean = dot(c, x[i]), precision = γ)
        x_prev = x[i]
    end
end

@constraints function conjugate_ar_constraints()
    q(x0, x, w) = q(x0, x)q(w)
end

@algorithm function conjugate_ar_algorithm(order)
    ConjugateAR() -> ARVMP(Multivariate, order, ARsafe())
end
```

## Limitations

- A model must give [`ARVMP`](@ref), with the `Multivariate` form; under `Univariate` the rules
  raise a `MethodError`.
- The rule towards `w` and the average energy need the structured factorisation
  `q(y, x) q(w)`: under the mean field `q(y) q(x) q(w)` there is no message towards `w` and no
  average energy.
- The message towards `w` is improper for `p ≥ 3`, its shape `(3 - p)/2` being non-positive;
  its product with a proper normal-gamma prior is proper. `q(w)` therefore needs a prior, and an
  initial marginal where the schedule reads it first.

## API

```@docs
ConjugateAR
```
