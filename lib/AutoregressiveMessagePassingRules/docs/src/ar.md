```@meta
DocTestSetup = :(using AutoregressiveMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily)
```

# [AR](@id ar-node)

## Overview

[`AR`](@ref) is the autoregressive process of order `p`, whose current value is a linear
combination of its `p` previous values plus Gaussian noise. It is the building block of latent
autoregressive models, in which the states, the coefficients `θ` and the noise precision `γ`
are all inferred, each with a prior of its own: a normal on `θ`, a gamma on `γ`. Chained over
time, `x[t] ~ AR(x[t - 1], θ, γ)`, it is a state-space model whose transition is learnt.

## Definition

For the state `xₜ = (yₜ₋₁, …, yₜ₋ₚ)`:

```math
p(y_t \mid x_t, \theta, \gamma) = \mathcal{N}\big(y_t \mid \theta^\top x_t, \, \gamma^{-1}\big).
```

For `p > 1` the edges carry whole states: `y = (yₜ, …, yₜ₋ₚ₊₁)`, related to `x` by the
companion matrix `A(θ)`, whose first row is `θᵀ` and whose sub-diagonal is ones, with noise on
the first component only:

```math
y = A(\theta)\, x + \varepsilon, \qquad
A(\theta) = \begin{pmatrix} \theta_1 & \theta_2 & \cdots & \theta_{p-1} & \theta_p \\ 1 & 0 & \cdots & 0 & 0 \\ & \ddots & & & \vdots \\ 0 & 0 & \cdots & 1 & 0 \end{pmatrix}, \qquad
\varepsilon \sim \mathcal{N}\big(0, \operatorname{diag}(\gamma^{-1}, 0, \dots, 0)\big).
```

The other components of `y` are copies of those of `x`, so the node's density is degenerate in
them; the rules and the average energy account for that.

## Interfaces

| name | aliases | meaning | messages and marginals its rules take |
|---|---|---|---|
| `y` | `out` | the current state | a normal message; `q(y)` of any type with a mean and covariance |
| `x` | | the previous state | a normal message; `q(x)` of any type with a mean and covariance |
| `θ` | | the coefficients, of dimension `p` | `q(θ)` with a mean and covariance, a normal in practice |
| `γ` | | the precision of the noise | `q(γ)` with `mean` and `mean(log, ·)`, a gamma in practice |

Under `ARVMP(Univariate, 1, …)` every edge is a scalar; under `ARVMP(Multivariate, p, …)` `y`,
`x` and `θ` are vectors of length `p`, even for `p = 1`. Messages of the other form
raise a `MethodError`. The joint marginal `q(y, x)` is a multivariate normal of dimension `2p`,
`y` first.

## Algorithm

The node's rules run under [`ARVMP`](@ref)`(form, order, stype)`, which the model must name:
the node declares no algorithm of its own, and under
[`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm) it has no rule. All three
arguments are required, without defaults:

- `form`: `Univariate` for an AR(1) with scalar edges, or `Multivariate`;
- `order`: the order `p`; `Univariate` forces it to `1`, with a warning for any other value;
- `stype`: [`ARsafe`](@ref)`()`, which forms the joint `q(y, x)` from its precision,
  regularising the noiseless components with a large finite precision, or
  [`ARunsafe`](@ref)`()`, which forms it from its covariance, exactly, through the Kalman gain.
  The two agree up to that regularisation, and exactly for a univariate AR(1).

## Supported rules

```@example
using MessagePassingRulesBase, AutoregressiveMessagePassingRules # hide
MessagePassingRulesBase.rule_coverage(AR)
```

Every rule is under [`ARVMP`](@ref), none under the default algorithm. Each target has two
rules, one for each factorisation: the structured `q(y, x) q(θ) q(γ)`, in which `y` and `x`
exchange messages by belief propagation and the rules towards `θ` and `γ` read the joint
`q(y, x)`, and the mean field `q(y) q(x) q(θ) q(γ)`, in which every rule reads marginals only.
The joint marginal `q(y, x)` has a rule from the messages on `y` and `x`, and both
factorisations have an average energy, so the node contributes to the Bethe free energy.

## Example

The message towards `γ` of an AR(2), from the joint marginal of the states and the marginal of
the coefficients: a gamma of shape `3/2` whose rate is half the expected squared residual.

```jldoctest
julia> result = @call_message_update_rule(
           node = AR, target = :γ, algorithm = ARVMP(Multivariate, 2, ARsafe()),
           clusters = ((:y, :x) => MvNormalMeanCovariance(ones(4), [1.0 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]),),
           q = (θ = MvNormalMeanCovariance([0.5, 0.25], [1.0 0.0; 0.0 1.0]),),
       );

julia> shape(getresult(result)) ≈ 1.5 && rate(getresult(result)) ≈ 2.6875
true
```

## Limitations

- A model must give [`ARVMP`](@ref); without it the node has no rule.
- The rules are variational only: there is no belief propagation towards `θ` or `γ`, so both
  need an initial marginal, as does any edge a mean-field schedule reads first.
- The messages on `y` and `x` must be normals of the algorithm's variate form.
- The mean-field rule towards `γ` computes the expected squared residual without the term
  `tr(Vθ Vx)`, the product of the coefficients' and the previous state's covariances; the
  structured rule includes it.

## API

The node and its alias:

```@docs
AR
Autoregressive
```

Its algorithm, and the two ways it forms the joint `q(y, x)`:

```@docs
ARVMP
ARsafe
ARunsafe
```
