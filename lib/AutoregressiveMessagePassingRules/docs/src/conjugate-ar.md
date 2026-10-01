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

Here `q(w)` is the [marginal](@extref MessagePassingRulesBase glossary-marginal) of `w`, the
approximate posterior that message passing computes. As for [`AR`](@ref), the node's
[rules](@extref MessagePassingRulesBase glossary-rule) are those of
[variational message passing](@extref MessagePassingRulesBase glossary-vmp).

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

that is `Λ = C`, `μ = C⁻¹ b`, shape `(3 - p)/2` and rate `(a - bᵀ C⁻¹ b)/2`. These are the
parameters of the [message](@extref MessagePassingRulesBase glossary-message) towards `w`.

## Interfaces

```@example conjugate-ar
using AutoregressiveMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily
MessagePassingRulesBase.nodespec(ConjugateAR)
```

The node has three [interfaces](@extref MessagePassingRulesBase glossary-interface); `y` also
answers to `out`.

| name | aliases | meaning | messages and marginals its rules take |
|---|---|---|---|
| `y` | `out` | the current state | a normal message, multivariate unless the form is `Univariate`; `q(y)` with a mean and covariance |
| `x` | | the previous state | a normal message of the same dimension; `q(x)` with a mean and covariance |
| `w` | | the coefficients and the precision | `q(w)`, an `MvNormalGamma` of dimension `p` |

The joint marginal `q(y, x)` is a multivariate normal of dimension `2p`, `y` first.

## Algorithm

The node shares [`AR`](@ref)'s [algorithm](@extref MessagePassingRulesBase glossary-algorithm),
[`ARVMP`](@ref)`(form, order, stype)`, which the model must name, with every argument required.
`form` is `Univariate` for an AR(1) on scalars, whose rules read `q(w)`'s one coefficient as a
scalar `θ`, or `Multivariate` for an AR of any order on state vectors. `stype` is
[`ARsafe`](@ref)`()` or [`ARunsafe`](@ref)`()`, as for [`AR`](@ref). The node declares no
[dependencies](@extref MessagePassingRulesBase glossary-dependencies), so each rule takes the
inputs of the [default scheme](@extref MessagePassingRulesBase glossary-default-scheme).

## Supported rules

```@example conjugate-ar
MessagePassingRulesBase.rule_coverage(ConjugateAR)
```

Every rule is under [`ARVMP`](@ref). The rules towards `y` and `x` exist for the
[structured](@extref MessagePassingRulesBase glossary-structured-vmp)
[factorisation](@extref MessagePassingRulesBase glossary-factorisation) `q(y, x) q(w)`, as
[belief propagation](@extref MessagePassingRulesBase glossary-belief-propagation) between them,
and for the [mean field](@extref MessagePassingRulesBase glossary-mean-field). They and the joint
marginal `q(y, x)` are [`AR`](@ref)'s, computed from the marginals of `θ` and `γ` that `q(w)`
implies. The rule towards `w` and the
[average energy](@extref MessagePassingRulesBase glossary-average-energy) need the structured
factorisation `q(y, x) q(w)`.

## Example

The message towards `w` of an AR(1) from the joint marginal of the states:

```@example conjugate-ar
@call_message_update_rule(
    node = ConjugateAR, target = :w, algorithm = ARVMP(Multivariate, 1, ARsafe()),
    clusters = ((:y, :x) => MvNormalMeanCovariance([1.0, 0.5], [1.0 0.0; 0.0 1.0]),),
)
```

[`@call_message_update_rule`](@extref MessagePassingRulesBase.@call_message_update_rule) runs
the rule and draws the joint marginal it read. With `C = 1.25`, `b = 0.5` and `a = 2`, the
[`getresult`](@extref MessagePassingRulesBase.getresult) of the call is the normal-gamma of the
definition:

```jldoctest
julia> result = @call_message_update_rule(
           node = ConjugateAR, target = :w, algorithm = ARVMP(Multivariate, 1, ARsafe()),
           clusters = ((:y, :x) => MvNormalMeanCovariance([1.0, 0.5], [1.0 0.0; 0.0 1.0]),),
       );

julia> μ, Λ, α, β = params(getresult(result));

julia> μ ≈ [0.4] && Λ ≈ fill(1.25, 1, 1) && α ≈ 1.0 && β ≈ 0.9
true
```

## Limitations

- A model must give [`ARVMP`](@ref); without it the node has no rule.
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
