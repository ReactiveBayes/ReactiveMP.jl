# [MultinomialPolya](@id page-multinomial-polya)

## Overview

[`MultinomialPolya`](@ref) is a multinomial over `K` categories whose probabilities come from `K - 1`
log-odds `ψ` by logistic stick-breaking: the first category takes `σ(ψ₁)` of the stick, the
second `σ(ψ₂)` of what is left, and the last the rest. The multinomial then factors into `K - 1`
binomials, each Pólya-Gamma augmented as [the overview](@ref polya-augmentation) shows, so
the [message](@extref MessagePassingRulesBase glossary-message) towards `ψ` is normal. With `ψ` a
linear function of covariates it is multinomial regression.

## Definition

```math
p(x \mid N, \psi) = \frac{N!}{\prod_{k=1}^{K} x_k!} \prod_{k=1}^{K} p_k^{x_k}, \qquad
p_k = \sigma(\psi_k) \prod_{j<k} \big(1 - \sigma(\psi_j)\big), \quad
p_K = \prod_{j<K} \big(1 - \sigma(\psi_j)\big)
```

The first factor is the multinomial coefficient; `p_k` is the share of the stick that category
`k` takes. The `k`-th binomial is of `x_k` out of `N_k = N - Σ_{j<k} x_j` trials, with log-odds
`ψ_k`. [`logistic_stick_breaking`](@ref) computes the `p_k` and [`compose_Nks`](@ref) the `N_k`.

## Interfaces

```@example multinomial
using PolyaMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily
MessagePassingRulesBase.nodespec(MultinomialPolya)
```

The node has three [interfaces](@extref MessagePassingRulesBase glossary-interface). The table
lists the [marginals](@extref MessagePassingRulesBase glossary-marginal) and messages the rules
read on each.

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

[`MultinomialPolyaApproximation`](@ref)`()` is the node's default
[algorithm](@extref MessagePassingRulesBase glossary-algorithm), so a model names it only to
change the number of cubature points of the average energy.

The node declares its [dependencies](@extref MessagePassingRulesBase glossary-dependencies), the
inputs each target's rule takes:

```@example multinomial
MessagePassingRulesBase.dependencies_spec(MultinomialPolya, MultinomialPolyaApproximation())
```

Every target takes the inputs of the
[default scheme](@extref MessagePassingRulesBase glossary-default-scheme). The rule towards `ψ`
also reads the message on `ψ`, the current estimate at which it takes the Pólya-Gamma means.

## Supported rules

```@example multinomial
MessagePassingRulesBase.rule_coverage(MultinomialPolya)
```

The rules towards `x` and `ψ` read marginals and, towards `ψ`, the message on `ψ`. They fit a
[mean-field](@extref MessagePassingRulesBase glossary-mean-field)
[factorisation](@extref MessagePassingRulesBase glossary-factorisation). With the
[average energy](@extref MessagePassingRulesBase glossary-average-energy), the
[Bethe free energy](@extref MessagePassingRulesBase glossary-bethe-free-energy) is available.

## Example

The message towards `ψ` for ten trials over three categories, at a message on `ψ` centred on
zero:

```@example multinomial
@call_message_update_rule(
    node = MultinomialPolya, target = :ψ,
    q = (x = PointMass([2, 3, 5]), N = PointMass(10)),
    m = (ψ = MvNormalWeightedMeanPrecision(zeros(2), [1.0 0.0; 0.0 1.0]),),
)
```

[`@call_message_update_rule`](@extref MessagePassingRulesBase.@call_message_update_rule) runs
the rule and draws its inputs. The stick's two breaks leave `N_k = 10` and `8` trials, so the
weighted means are `x_k - N_k/2` and the precisions `N_k/4`, which
[`getresult`](@extref MessagePassingRulesBase.getresult) confirms:

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

## Limitations

- The rule towards `ψ` needs an **[initial message](@extref MessagePassingRulesBase glossary-initial-message) on `ψ`**.
- There is no rule towards `N`, and the average energy takes a `PointMass` `q(N)` only.
- The messages use the mean of `ψ` only: its variance enters the average energy, not them.
- The average energy computes each `⟨softplus(ψ_k)⟩` by Gauss–Hermite cubature with
  `MultinomialPolyaApproximation(points = …)` points, 21 by default. It is exact to rounding
  while `ψ_k` is narrow, a variance up to about 1, and loses accuracy as it broadens, since
  softplus bends sharply on the
  cubature's scale: about `1e-4` relative at a variance of 25 and `1e-2` at 400, and more
  points help slowly. The free energy of a model with broad priors is approximate in its first
  iterations; the messages do not use the cubature.

## API

The node, and the two exported helpers its rules are built on: the stick-breaking
probabilities and the trials left at each break.

```@docs
MultinomialPolya
logistic_stick_breaking
compose_Nks
```
