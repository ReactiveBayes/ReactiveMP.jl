# SoftDotMessagePassingRules

The [`SoftDot`](@ref) node and its rules: a dot product of two unknowns, softened by Gaussian
noise so that every message is in closed form. Use it for a Bayesian linear regression,
`y ~ N(θᵀx, γ⁻¹)` with the noise precision `γ` learnt, or wherever a dot product joins two
Gaussian unknowns, which the deterministic dot product node of StandardMessagePassingRules cannot
do in closed form. This site is a single page.

```@docs
SoftDotMessagePassingRules
```

!!! info "Where these rules run"
    [ReactiveMP](https://reactivebayes.github.io/ReactiveMP.jl/dev/) runs these rules on a factor
    graph, which [RxInfer](https://github.com/ReactiveBayes/RxInfer.jl) builds from a
    [GraphPPL](https://github.com/ReactiveBayes/GraphPPL.jl) model. The examples here call the
    rules directly, as a test does.

## Overview

A deterministic node `y = θᵀx` with both `θ` and `x` unknown has no closed-form
[messages](@extref MessagePassingRulesBase glossary-message). `SoftDot` replaces the constraint
by a Gaussian likelihood of precision `γ` around it. The softening makes it a
[stochastic node](@extref MessagePassingRulesBase glossary-stochastic-node), a density over its
variables.

Under [variational message passing](@extref MessagePassingRulesBase glossary-vmp), the node then
sends a normal message towards each of `y`, `θ` and `x` and a gamma message towards `γ`, all in
closed form. It contributes an
[average energy](@extref MessagePassingRulesBase glossary-average-energy) to the
[Bethe free energy](@extref MessagePassingRulesBase glossary-bethe-free-energy). A large `γ`, or
a prior on `γ` concentrated at large values, brings it close to the deterministic dot product.

## Definition

```math
p(y \mid θ, x, γ) = \mathcal{N}\left(y \mid θ^\top x, γ^{-1}\right)
= \sqrt{\frac{γ}{2π}} \exp\left(-\frac{γ}{2} \left(y - θ^\top x\right)^2\right)
```

The output `y` is normal around the dot product `θᵀx`, with variance `γ⁻¹`. `y` is a scalar;
`θ` and `x` are both scalars or both vectors of the same length.

## Interfaces

```@example softdot
using SoftDotMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily
MessagePassingRulesBase.nodespec(SoftDot)
```

The node has four [interfaces](@extref MessagePassingRulesBase glossary-interface), two of them
with an alias that a model may write instead of the Greek name. The node itself has the alias
[`softdot`](@ref).

| name | alias | meaning | its rules read |
|:-----|:------|:--------|:---------------|
| `y` | | the result of the soft dot product, a scalar | `q(y)` a univariate normal; under `q(y, x)`, the message on `y`, a univariate normal |
| `θ` | `theta` | the first factor | `q(θ)` a normal, univariate or multivariate |
| `x` | | the second factor | `q(x)` a normal, or a point mass for known regressors; under `q(y, x)`, the message on `x`, a normal |
| `γ` | `gamma` | the precision of the noise | `q(γ)` a gamma, through its mean (and the mean of its logarithm for the energy) |

Here `q(θ)` is the [marginal](@extref MessagePassingRulesBase glossary-marginal) of `θ`. The
rules towards `θ` and `x` return a normal in weighted-mean–precision form, univariate or
multivariate as the factor is. The rule towards `y` returns a univariate normal and the rule
towards `γ` a `GammaShapeRate`. Under the structured factorisation the joint `q(y, x)` is a
multivariate normal over `[y; x]`.

## Algorithm

`SoftDot` runs under [`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm),
which takes no parameters; a model need not name it. The node declares no
[dependencies](@extref MessagePassingRulesBase glossary-dependencies), so each rule takes the
inputs of the [default scheme](@extref MessagePassingRulesBase glossary-default-scheme).

## Supported rules

```@example softdot
MessagePassingRulesBase.rule_coverage(SoftDot)
```

[`rule_coverage`](@extref MessagePassingRulesBase.rule_coverage) counts two rules per target,
one per [factorisation](@extref MessagePassingRulesBase glossary-factorisation):

- the [mean field](@extref MessagePassingRulesBase glossary-mean-field) `q(y)q(θ)q(x)q(γ)`,
  where every rule reads the marginals of the other interfaces;
- the [structured](@extref MessagePassingRulesBase glossary-structured-vmp) `q(y, x)q(θ)q(γ)`,
  where the rules towards `y` and `x` read the other one's message and the rules towards `θ`
  and `γ` read the joint `q(y, x)`.

The marginal rule `q(y, x)` computes that joint from the messages on `y` and `x`. There are two
average energies, one per factorisation. There are no
[belief propagation](@extref MessagePassingRulesBase glossary-belief-propagation) rules.

## Example

The message towards `γ` under mean field is `Γ(3/2, ⟨(y - θᵀx)²⟩ / 2)`. With every marginal
`N(1, 1)`, the expected square is `E[y²] - 2E[y]E[θ]E[x] + E[θ²]E[x²] = 2 - 2 + 4`:

```@example softdot
@call_message_update_rule(
    node = SoftDot, target = :γ,
    q = (y = NormalMeanVariance(1.0, 1.0), θ = NormalMeanVariance(1.0, 1.0), x = NormalMeanVariance(1.0, 1.0)),
)
```

[`@call_message_update_rule`](@extref MessagePassingRulesBase.@call_message_update_rule) runs
the rule and draws the marginals it read. The result is `Γ(1.5, 2.0)`, which
[`getresult`](@extref MessagePassingRulesBase.getresult) extracts:

```jldoctest
julia> using MessagePassingRulesBase, SoftDotMessagePassingRules, ExponentialFamily, BayesBase

julia> q = (y = NormalMeanVariance(1.0, 1.0), θ = NormalMeanVariance(1.0, 1.0), x = NormalMeanVariance(1.0, 1.0));

julia> message = getresult(@call_message_update_rule(node = SoftDot, target = :γ, q = q));

julia> shape(message) ≈ 1.5 && rate(message) ≈ 2.0
true
```

## Limitations

- Variational rules only: there is no belief propagation rule, so the node needs a
  factorisation, mean-field or `q(y, x)q(θ)q(γ)`. It also needs initial marginals where the
  schedule reads them before they are computed.
- No rules for other factorisations, such as a joint over `θ` and `x` or over `y` and `θ`.
- `y` is a scalar: a vector-valued output needs one node per component.
- The rules take normal marginals and messages on `y`, `θ` and `x` and a gamma-like marginal on
  `γ` (anything with a mean, and a mean of the logarithm for the energy).

## API

The node and its alias, the name an RxInfer model writes:

```@docs
SoftDot
softdot
```
