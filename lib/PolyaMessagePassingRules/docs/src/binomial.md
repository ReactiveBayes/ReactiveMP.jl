# [BinomialPolya](@id page-binomial-polya)

## Overview

`BinomialPolya` is a binomial regression through the logistic: `y` successes out of `n` trials,
with success probability `σ(xᵀβ)` for the covariates `x` and the weights `β`. With `n = 1` it is
logistic regression. `y`, `x` and `n` are observed; the node learns `β`, whose message is normal
by the Pólya-Gamma augmentation.

## Definition

```math
p(y \mid x, n, \beta) = \binom{n}{y} \sigma(x^\top \beta)^{y} \big(1 - \sigma(x^\top \beta)\big)^{n - y},
\qquad \sigma(\psi) = \frac{1}{1 + e^{-\psi}}
```

## Interfaces

| name | meaning | messages and marginals the rules take |
|---|---|---|
| `y` | the number of successes | `PointMass` marginal |
| `x` | the covariates, a vector, or a number for a scalar `β` | `PointMass` marginal |
| `n` | the number of trials | `PointMass` marginal (any marginal with a mean, towards `y`) |
| `β` | the weights | normal message on `β` towards `β`; normal marginal towards `y` and for the energy |

The rule towards `β` sends a normal in weighted-mean and precision form, `MvNormal` for a vector
`x`; the rule towards `y` sends a `Binomial`. There is no rule towards `x` or `n`.

## Algorithm

```@docs
BinomialPolyaApproximation
```

`BinomialPolyaApproximation()` is the node's default algorithm, so a model names it only to
sample, as `BinomialPolyaApproximation(samples = 100)`. Sampling draws from the rule context's
generator, which the engine supplies.

## Supported rules

```@example
using MessagePassingRulesBase, PolyaMessagePassingRules # hide
MessagePassingRulesBase.rule_coverage(BinomialPolya)
```

The three columns are the algorithm's variants: `BinomialPolyaApproximation{Nothing}`, the means
(the default), `BinomialPolyaApproximation{Int}`, sampling, and the type itself, under which the
average energy serves both. The rules read marginals of the observed interfaces and, towards
`β`, the message on `β`: they fit a mean-field factorisation, in which `y`, `x` and `n` are data.
With the average energy, the Bethe free energy is available.

## Example

The message towards `β` for one observation, at a message on `β` centred on zero, where the
Pólya-Gamma mean is `n / 4`:

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

In a model, with RxInfer, the message on `β` is initialised:

```julia
using RxInfer, PolyaMessagePassingRules

@model function binomial_regression(y, X, n, d)
    β ~ MvNormal(mean = zeros(d), covariance = diageye(d))
    for i in eachindex(y)
        y[i] ~ BinomialPolya(X[i], n[i], β)
    end
end

@initialization function binomial_init(d)
    μ(β) = MvNormalWeightedMeanPrecision(zeros(d), diageye(d))
end

result = infer(
    model = binomial_regression(d = 2),
    data = (y = y, X = X, n = n),
    initialization = binomial_init(2),
    iterations = 20,
    free_energy = true,
)
```

## Limitations

- The rule towards `β` needs an **initial message on `β`**.
- `y`, `x` and `n` must be **observed**: there are no rules towards `x` or `n`, and the average
  energy takes `PointMass` marginals of all three.
- The average energy uses a fixed 32-point Gauss–Hermite cubature, which no keyword changes.

## API

The node:

```@docs
BinomialPolya
```
