# [MultinomialPolya](@id page-multinomial-polya)

## Overview

`MultinomialPolya` is a multinomial over `K` categories whose probabilities come from `K - 1`
log-odds `ψ` by logistic stick-breaking: the first category takes `σ(ψ₁)` of the stick, the
second `σ(ψ₂)` of what is left, and the last the rest. The multinomial then factors into `K - 1`
binomials, each Pólya-Gamma augmented, so the message towards `ψ` is normal. With `ψ` a linear
function of covariates it is multinomial regression.

## Definition

```math
p(x \mid N, \psi) = \frac{N!}{\prod_{k=1}^{K} x_k!} \prod_{k=1}^{K} p_k^{x_k}, \qquad
p_k = \sigma(\psi_k) \prod_{j<k} \big(1 - \sigma(\psi_j)\big), \quad
p_K = \prod_{j<K} \big(1 - \sigma(\psi_j)\big)
```

The `k`-th binomial is of `x_k` out of `N_k = N - Σ_{j<k} x_j` trials, with log-odds `ψ_k`.

## Interfaces

| name | meaning | messages and marginals the rules take |
|---|---|---|
| `x` | the counts of the `K` categories | any marginal with a vector mean, `PointMass` or `Multinomial` for the energy |
| `N` | the number of trials | `PointMass`, `Poisson`, `Binomial` or `Categorical`, by its mode; `PointMass` for the energy |
| `ψ` | the `K - 1` log-odds | normal message towards `ψ`; any marginal with a mean towards `x`; normal or `PointMass` for the energy |

The rule towards `ψ` sends a normal in weighted-mean and precision form, with a diagonal
precision, univariate when `K = 2`; the rule towards `x` sends a `Multinomial`. There is no rule
towards `N`.

## Algorithm

```@docs
MultinomialPolyaApproximation
```

`MultinomialPolyaApproximation()` is the node's default algorithm, so a model names it only to
change the number of cubature points of the average energy.

## Supported rules

```@example
using MessagePassingRulesBase, PolyaMessagePassingRules # hide
MessagePassingRulesBase.rule_coverage(MultinomialPolya)
```

The rules towards `x` and `ψ` read marginals and, towards `ψ`, the message on `ψ`: they fit a
mean-field factorisation. With the average energy, the Bethe free energy is available.

## Example

The message towards `ψ` for ten trials over three categories, at a message on `ψ` centred on
zero:

```jldoctest
julia> using PolyaMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily

julia> result = @call_message_update_rule(
           node = MultinomialPolya, target = :ψ,
           q = (x = PointMass([2, 3, 5]), N = PointMass(10)),
           m = (ψ = MvNormalWeightedMeanPrecision(zeros(2), [1.0 0.0; 0.0 1.0]),),
       );

julia> weightedmean(getresult(result)) ≈ [2 - 10 / 2, 3 - 8 / 2]
true

julia> precision(getresult(result)) ≈ [10/4 0.0; 0.0 8/4]
true
```

In a model, with RxInfer, the message on `ψ` is initialised:

```julia
using RxInfer, PolyaMessagePassingRules

@model function multinomial_model(x, N, K)
    ψ ~ MvNormal(mean = zeros(K - 1), covariance = diageye(K - 1))
    for i in eachindex(x)
        x[i] ~ MultinomialPolya(N, ψ)
    end
end

@initialization function multinomial_init(K)
    μ(ψ) = MvNormalWeightedMeanPrecision(zeros(K - 1), diageye(K - 1))
end
```

## Limitations

- The rule towards `ψ` needs an **initial message on `ψ`**.
- There is no rule towards `N`, and the average energy takes a `PointMass` `q(N)` only.
- The messages use the mean of `ψ` only: its variance enters the average energy, not them.

## API

The node, and the two exported helpers its rules are built on: the stick-breaking
probabilities and the trials left at each break.

```@docs
MultinomialPolya
logistic_stick_breaking
compose_Nks
```
