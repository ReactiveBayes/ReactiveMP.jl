# Gamma, Beta and discrete distributions

The positive, unit-interval and discrete distribution nodes. They are the types of
ExponentialFamily and Distributions, declared as nodes here, and all run under
[`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm). Many of them are
priors: they have the message towards `out` and no rule towards their parameters, which a model
then gives as constants. The coverage tables show which.

```@setup gbd
using MessagePassingRulesBase, StandardMessagePassingRules, ExponentialFamily, Distributions
```

## Example

An observed coin flip sends its bias a Beta likelihood, whose log scale is `log(1/2)`:

```jldoctest gbd
julia> using StandardMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase

julia> result = @call_message_update_rule(node = Bernoulli, target = :p, m = (out = PointMass(1.0),));

julia> getresult(result) isa Beta && all(params(getresult(result)) .≈ (2.0, 1.0))
true

julia> getlogscale(result) ≈ log(1 / 2)
true
```

## GammaShapeRate

```math
p(\mathrm{out} \mid α, β) = \frac{β^α}{Γ(α)} \mathrm{out}^{α - 1} e^{-β \mathrm{out}}, \quad \mathrm{out} > 0
```

| interface | aliases | meaning | messages and marginals its rules take |
|---|---|---|---|
| `out` | | the variable | any marginal with `E[out]`, `E[log out]` |
| `α` | `a`, `shape` | the shape | a `PointMass` message, or any marginal with `E[α]` |
| `β` | `b`, `rate` | the rate | a `PointMass` message, or a Gamma marginal |

```@example gbd
MessagePassingRulesBase.rule_coverage(GammaShapeRate)
```

The rules towards `α` and `β` are variational: towards `β` a `GammaShapeRate`, towards `α` a
[`GammaShapeLikelihood`](@ref), which needs a Gamma marginal of `β`. The average energy takes a
`PointMass` or a Gamma marginal of `α`. `Gamma` and `GammaShapeRate` are different nodes: this
one is parametrised by the rate.

## Gamma

```math
p(\mathrm{out} \mid α, θ) = \frac{\mathrm{out}^{α - 1} e^{-\mathrm{out} / θ}}{Γ(α) θ^α}, \quad \mathrm{out} > 0
```

| interface | aliases | meaning | messages and marginals its rules take |
|---|---|---|---|
| `out` | | the variable | any marginal (average energy) |
| `α` | `shape` | the shape | a `PointMass` message, or any marginal |
| `θ` | `scale` | the scale | a `PointMass` message, or any marginal |

```@example gbd
MessagePassingRulesBase.rule_coverage(Gamma)
```

**Limitations.** No rules towards `α` or `θ`. The variational message towards `out` is
`Gamma(E[α], E[θ])`, with the mean of the scale.

## GammaInverse

`GammaInverse` is ExponentialFamily's name for Distributions' `InverseGamma`, which is why the
table names it so.

```math
p(\mathrm{out} \mid α, θ) = \frac{θ^α}{Γ(α)} \mathrm{out}^{-α - 1} e^{-θ / \mathrm{out}}, \quad \mathrm{out} > 0
```

| interface | aliases | meaning | messages and marginals its rules take |
|---|---|---|---|
| `out` | | the variable | a `GammaInverse` marginal (average energy) |
| `α` | `shape` | the shape | a `PointMass` message, or any marginal |
| `θ` | `scale` | the scale | a `PointMass` message, or any marginal |

```@example gbd
MessagePassingRulesBase.rule_coverage(GammaInverse)
```

**Limitations.** No rules towards `α` or `θ`; the average energy needs both known.

## Beta

```math
p(\mathrm{out} \mid a, b) = \frac{\mathrm{out}^{a - 1} (1 - \mathrm{out})^{b - 1}}{B(a, b)}, \quad 0 < \mathrm{out} < 1
```

| interface | aliases | meaning | messages and marginals its rules take |
|---|---|---|---|
| `out` | | the variable | a `Beta` message (marginal rule), any marginal (average energy) |
| `a` | `α` | the first shape | a `PointMass` |
| `b` | `β` | the second shape | a `PointMass` |

```@example gbd
MessagePassingRulesBase.rule_coverage(Beta)
```

**Limitations.** No rules towards `a` or `b`. The average energy accepts any marginals of `a`
and `b` but uses only their means, which is exact for known shapes.

## Bernoulli

```math
p(\mathrm{out} \mid p) = p^{\mathrm{out}} (1 - p)^{1 - \mathrm{out}}, \quad \mathrm{out} \in \{0, 1\}
```

| interface | aliases | meaning | messages and marginals its rules take |
|---|---|---|---|
| `out` | | the outcome | a `PointMass` message; a `PointMass`, `Bernoulli` or two-category `Categorical` marginal |
| `p` | `θ` | the probability of 1 | a `Beta` or `PointMass` message; any marginal with `E[log p]`, `E[log(1 - p)]` |

```@example gbd
MessagePassingRulesBase.rule_coverage(Bernoulli)
```

Towards `p` the message is the likelihood `Beta(1 + r, 2 - r)` of an outcome `r`; towards
`out`, a `Bernoulli`. A `Categorical` marginal of `out` with other than two categories is an
`ArgumentError`.

## Categorical

```math
p(\mathrm{out} \mid p) = \prod_{k=1}^K p_k^{\mathrm{out}_k}, \quad \mathrm{out} \text{ one-hot}
```

| interface | aliases | meaning | messages and marginals its rules take |
|---|---|---|---|
| `out` | | the one-hot outcome | a `Categorical` or one-hot vector `PointMass` marginal |
| `p` | | the probability vector | a `Dirichlet` or `PointMass`, as message or marginal |

```@example gbd
MessagePassingRulesBase.rule_coverage(Categorical)
```

**Limitations.** The rules towards `p` take the marginal of `out` only, a `Categorical` or a
one-hot `PointMass`, and give a `Dirichlet`; any other marginal, or a `PointMass` that is not
one-hot, is an `ArgumentError`. There is no rule towards `p` from a message on `out`. The
variational message towards `out` keeps every probability above `tiny`.

## Dirichlet

```math
p(\mathrm{out} \mid a) = \frac{Γ(\sum_k a_k)}{\prod_k Γ(a_k)} \prod_{k=1}^K \mathrm{out}_k^{a_k - 1}
```

| interface | aliases | meaning | messages and marginals its rules take |
|---|---|---|---|
| `out` | | the probability vector | a `Dirichlet` |
| `a` | | the concentrations | a vector `PointMass` |

```@example gbd
MessagePassingRulesBase.rule_coverage(Dirichlet)
```

**Limitations.** No rule towards `a`, which must be known.

## DirichletCollection

A collection of independent Dirichlets, one for each slice along the first dimension of `out`
and of `a`:

```math
p(\mathrm{out} \mid a) = \prod_{j} \mathrm{Dir}(\mathrm{out}_{:, j} \mid a_{:, j})
```

| interface | aliases | meaning | messages and marginals its rules take |
|---|---|---|---|
| `out` | | the array of probability vectors | a `DirichletCollection` |
| `a` | | the array of concentrations | a `PointMass` |

```@example gbd
MessagePassingRulesBase.rule_coverage(DirichletCollection)
```

**Limitations.** No rule towards `a`, which must be known.

## Poisson

```math
p(\mathrm{out} \mid l) = \frac{l^{\mathrm{out}} e^{-l}}{\mathrm{out}!}, \quad \mathrm{out} \in \{0, 1, 2, \dots\}
```

| interface | aliases | meaning | messages and marginals its rules take |
|---|---|---|---|
| `out` | | the count | a `PointMass` message, or any marginal with `E[out]` |
| `l` | `λ` | the rate | a `PointMass` or a `Gamma` message, or a Gamma marginal |

```@example gbd
MessagePassingRulesBase.rule_coverage(Poisson)
```

Towards `l` the message is the likelihood `Gamma(out + 1, 1)`; towards `out`, a `Poisson`.

**Limitations.** For an `out` that is not a point mass, the average energy approximates
`E[log out!]` by a series in `E[out]`, and is an error beyond `E[out] = 110`.

## Uniform

```math
p(\mathrm{out} \mid a, b) = \frac{1}{b - a}, \quad a \le \mathrm{out} \le b
```

| interface | aliases | meaning | messages and marginals its rules take |
|---|---|---|---|
| `out` | | the variable | a `Beta` marginal (average energy) |
| `a` | `α`, `left` | the lower bound | a `PointMass`, as message or marginal |
| `b` | `β`, `right` | the upper bound | a `PointMass`, as message or marginal |

```@example gbd
MessagePassingRulesBase.rule_coverage(Uniform)
```

**Limitations.** No rules towards `a` or `b`, and no marginal rule. The average energy is
defined only for `Uniform(0, 1)` with a `Beta` marginal of `out`, and is an `ArgumentError` for
other bounds. The product of `Uniform(0, 1)` with a `Beta` is the `Beta`, which the package
defines for the two foreign types; for any other bounds it is an `ArgumentError`.
