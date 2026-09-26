# SoftDotMessagePassingRules

The SoftDot node and its rules: a dot product of two unknowns, softened by Gaussian noise so that
every message is in closed form. Use it for a Bayesian linear regression, `y ~ N(θᵀx, γ⁻¹)` with
the noise precision `γ` learnt, or wherever a dot product joins two Gaussian unknowns, which the
deterministic dot product node of StandardMessagePassingRules cannot do in closed form. This site
is a single page.

```@docs
SoftDotMessagePassingRules
```

!!! info "Where these rules run"
    This package defines message passing rules; it does not build or run models. The
    [ReactiveMP](https://reactivebayes.github.io/ReactiveMP.jl/dev/) engine runs the rules on a
    factor graph, and [RxInfer](https://github.com/ReactiveBayes/RxInfer.jl) builds that graph from
    a model written with [GraphPPL](https://github.com/ReactiveBayes/GraphPPL.jl). The examples
    here call the rules directly, as a test or an interactive session does.

## Overview

`SoftDot` replaces the constraint `y = θᵀx` by a Gaussian likelihood of precision `γ` around it.
The softening makes the node stochastic, so variational message passing gives a normal message
towards each of `y`, `θ` and `x` and a gamma message towards `γ`, all in closed form, and the
node contributes an average energy to the Bethe free energy. A large `γ`, or a prior on `γ`
concentrated at large values, brings it close to the deterministic dot product.

## Definition

```math
p(y \mid θ, x, γ) = \mathcal{N}\left(y \mid θ^\top x, γ^{-1}\right)
= \sqrt{\frac{γ}{2π}} \exp\left(-\frac{γ}{2} \left(y - θ^\top x\right)^2\right)
```

`y` is a scalar; `θ` and `x` are both scalars or both vectors of the same length.

## Interfaces

| name | alias | meaning | its rules read |
|:-----|:------|:--------|:---------------|
| `y` | | the result of the soft dot product, a scalar | `q(y)` a univariate normal; under `q(y, x)`, the message on `y`, a univariate normal |
| `θ` | `theta` | the first factor | `q(θ)` a normal, univariate or multivariate |
| `x` | | the second factor | `q(x)` a normal, or a point mass for known regressors; under `q(y, x)`, the message on `x`, a normal |
| `γ` | `gamma` | the precision of the noise | `q(γ)` a gamma, through its mean (and the mean of its logarithm for the energy) |

The rules towards `θ` and `x` return a normal in weighted-mean–precision form, univariate or
multivariate as the factor is; the rule towards `y` returns a univariate normal and the rule
towards `γ` a `GammaShapeRate`. Under the structured factorisation the joint `q(y, x)` is a
multivariate normal over `[y; x]`.

## Algorithm

`SoftDot` runs under
[`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm), which takes no
parameters; a model need not name it.

## Supported rules

```@example
using MessagePassingRulesBase, SoftDotMessagePassingRules # hide
MessagePassingRulesBase.rule_coverage(SoftDot)
```

Every rule is variational. Each target has two: one for the mean-field factorisation
`q(y)q(θ)q(x)q(γ)`, reading the marginals of the other interfaces, and one for the structured
`q(y, x)q(θ)q(γ)`, where the rules towards `y` and `x` read the other one's message and the rules
towards `θ` and `γ` read the joint `q(y, x)`. The marginal rule `q(y, x)` computes that joint
from the messages on `y` and `x`. There are two average energies, one per factorisation. There
are no belief-propagation rules.

## Example

The message towards `γ` under mean-field is `Γ(3/2, ⟨(y - θᵀx)²⟩ / 2)`. With every marginal
`N(1, 1)`, the expected square is `E[y²] - 2E[y]E[θ]E[x] + E[θ²]E[x²] = 2 - 2 + 4`:

```jldoctest
julia> using MessagePassingRulesBase, SoftDotMessagePassingRules, ExponentialFamily, BayesBase

julia> q = (y = NormalMeanVariance(1.0, 1.0), θ = NormalMeanVariance(1.0, 1.0), x = NormalMeanVariance(1.0, 1.0));

julia> message = getresult(@call_message_update_rule(node = SoftDot, target = :γ, q = q));

julia> shape(message) ≈ 1.5 && rate(message) ≈ 2.0
true
```

## Limitations

- Variational rules only: there is no belief-propagation rule, so the node needs a
  factorisation, mean-field or `q(y, x)q(θ)q(γ)`, and initial marginals where the schedule reads
  them before they are computed.
- No rules for other factorisations, such as a joint over `θ` and `x` or over `y` and `θ`.
- `y` is a scalar: a vector-valued output needs one node per component.
- The rules take normal marginals and messages on `y`, `θ` and `x` and a gamma-like marginal on
  `γ` (anything with a mean, and a mean of the logarithm for the energy).

## API

The node and its alias, the name a model writes:

```@docs
SoftDot
softdot
```
