# [BinomialPolya](@id page-binomial-polya)

## Overview

[`BinomialPolya`](@ref) is a binomial regression through the logistic: `y` successes out of `n`
trials, with success probability `σ(xᵀβ)` for the covariates `x` and the weights `β`. With
`n = 1` it is logistic regression. `y`, `x` and `n` are observed; the node learns `β`, whose
[message](@extref MessagePassingRulesBase glossary-message) is normal by the Pólya-Gamma
[augmentation](@ref polya-augmentation).

## Definition

```math
p(y \mid x, n, \beta) = \binom{n}{y} \sigma(x^\top \beta)^{y} \big(1 - \sigma(x^\top \beta)\big)^{n - y},
\qquad \sigma(\psi) = \frac{1}{1 + e^{-\psi}}
```

The binomial coefficient counts the orders in which `y` successes can occur. The logistic
function `σ` maps the linear predictor `xᵀβ` to a probability.

## Interfaces

```@example binomial
using PolyaMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily
MessagePassingRulesBase.nodespec(BinomialPolya)
```

The node has four [interfaces](@extref MessagePassingRulesBase glossary-interface). The table
lists the [marginals](@extref MessagePassingRulesBase glossary-marginal) and messages the rules
read on each.

| name | meaning | messages and marginals the rules take |
|---|---|---|
| `y` | the number of successes | [`PointMass`](@extref MessagePassingRulesBase glossary-point-mass) marginal |
| `x` | the covariates, a vector, or a number for a scalar `β` | `PointMass` marginal |
| `n` | the number of trials | `PointMass` marginal (any marginal with a mean, towards `y`) |
| `β` | the weights | normal message on `β` towards `β`; normal marginal towards `y` and for the energy |

The rule towards `β` sends a normal in weighted-mean and precision form, `MvNormal` for a vector
`x`; the rule towards `y` sends a `Binomial`. There is no rule towards `x` or `n`.

## Algorithm

```@docs
BinomialPolyaApproximation
```

[`BinomialPolyaApproximation`](@ref)`()` is the node's default
[algorithm](@extref MessagePassingRulesBase glossary-algorithm), so a model names it only to
sample, as `BinomialPolyaApproximation(samples = 100)`. Sampling draws from the rule context's
generator, a [service](@extref MessagePassingRulesBase glossary-service) the engine supplies.

The node declares its [dependencies](@extref MessagePassingRulesBase glossary-dependencies), the
inputs each target's rule takes, for both variants of the algorithm:

```@example binomial
MessagePassingRulesBase.dependencies_spec(BinomialPolya, BinomialPolyaApproximation())
```

Every target takes the inputs of the
[default scheme](@extref MessagePassingRulesBase glossary-default-scheme). The rule towards `β`
also reads the message on `β`, the current estimate at which it takes the Pólya-Gamma mean.

## Supported rules

```@example binomial
MessagePassingRulesBase.rule_coverage(BinomialPolya)
```

The three columns are the algorithm's variants:

- `BinomialPolyaApproximation{Nothing}`, the means, the default;
- `BinomialPolyaApproximation{Int}`, sampling;
- the type itself, under which the
  [average energy](@extref MessagePassingRulesBase glossary-average-energy) serves both.

The rules read marginals of the observed interfaces and, towards `β`, the message on `β`. They
fit a [mean-field](@extref MessagePassingRulesBase glossary-mean-field)
[factorisation](@extref MessagePassingRulesBase glossary-factorisation), in which `y`, `x` and
`n` are data. With the average energy, the
[Bethe free energy](@extref MessagePassingRulesBase glossary-bethe-free-energy) is available.

## Example

The message towards `β` for one observation, at a message on `β` centred on zero, where the
Pólya-Gamma mean is `n / 4`:

```@example binomial
@call_message_update_rule(
    node = BinomialPolya, target = :β,
    q = (y = PointMass(3), x = PointMass([0.1, 0.2]), n = PointMass(5)),
    m = (β = MvNormalWeightedMeanPrecision(zeros(2), [1.0 0.0; 0.0 1.0]),),
)
```

[`@call_message_update_rule`](@extref MessagePassingRulesBase.@call_message_update_rule) runs
the rule and draws its inputs: the three observed marginals and the message on `β`, the target's
own edge. The weighted mean is `(y - n/2) x` and the precision `(n/4) x xᵀ`, which
[`getresult`](@extref MessagePassingRulesBase.getresult) confirms:

```jldoctest
julia> using PolyaMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily

julia> result = @call_message_update_rule(
           node = BinomialPolya, target = :β,
           q = (y = PointMass(3), x = PointMass([0.1, 0.2]), n = PointMass(5)),
           m = (β = MvNormalWeightedMeanPrecision(zeros(2), [1.0 0.0; 0.0 1.0]),),
       );

julia> weightedmean(getresult(result)) ≈ (3 - 5 / 2) * [0.1, 0.2]
true

julia> precision(getresult(result)) ≈ (5 / 4) * [0.1, 0.2] * [0.1, 0.2]'
true
```

## Limitations

- The rule towards `β` needs an **[initial message](@extref MessagePassingRulesBase glossary-initial-message) on `β`**.
- `y`, `x` and `n` must be **observed**: there are no rules towards `x` or `n`, and the average
  energy takes `PointMass` marginals of all three.
- The average energy uses a fixed 32-point Gauss–Hermite cubature, which no keyword changes.

## API

The node:

```@docs
BinomialPolya
```
