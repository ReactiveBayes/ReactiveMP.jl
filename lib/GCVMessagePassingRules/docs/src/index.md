# GCVMessagePassingRules

The Gaussian controlled variance node, [`GCV`](@ref), and its rules for reactive message passing:
a normal whose log-variance is linear in other variables of the model. Use it for a variance that is
itself inferred and changes over time, as in hierarchical Gaussian filters, where the state of one
layer sets the volatility of the layer below.

```@docs
GCVMessagePassingRules
```

!!! info "Where these rules run"
    The [ReactiveMP](https://reactivebayes.github.io/ReactiveMP.jl/dev/) engine runs these rules on
    a factor graph, which [RxInfer](https://github.com/ReactiveBayes/RxInfer.jl) builds from a model.
    The examples here call the rules directly.

## Overview

A [`GCV`](@ref) node relates an output `y` to its mean `x` with a variance `exp(κz + ω)`. Here `z`
is a latent variable, often a random walk of its own. `κ` scales the effect of `z` on the
log-variance, and `ω` is an offset.

The rules are [variational](@extref MessagePassingRulesBase glossary-vmp): each rule sends the
exponentiated expected log-density of the node, the expectation taken under the
[marginals](@extref MessagePassingRulesBase glossary-marginal) of the variables outside the
target's [cluster](@extref MessagePassingRulesBase glossary-cluster). Every rule
reads the marginals of `z`, `κ` and `ω`. The messages towards `z`, `κ` and `ω` are
[`ExponentialLinearQuadratic`](@ref) densities, whose moments a Gauss–Hermite cubature computes.

The package also adds rules to the
[`NormalMeanVariance`](@extref StandardMessagePassingRules NormalMeanVariance) and
[`NormalMeanPrecision`](@extref StandardMessagePassingRules NormalMeanPrecision) nodes of
StandardMessagePassingRules. These rules accept an `ExponentialLinearQuadratic` message on the
normal's `out`. With them, `z` or `ω` can itself be the output of a normal node, such as a step
of a random walk.

## Definition

```math
p(y \mid x, z, κ, ω) = \mathcal{N}\bigl(y \mid x, \exp(κ z + ω)\bigr)
= \frac{1}{\sqrt{2π \exp(κ z + ω)}} \exp\Bigl(-\frac{(y - x)^2}{2 \exp(κ z + ω)}\Bigr).
```

The rules need the expected noise precision `⟨exp(-(κz + ω))⟩ = ⟨exp(-ω)⟩ ⟨exp(-κz)⟩`. For a normal
`q(ω)` the first factor is the mean of a lognormal, `exp(-⟨ω⟩ + Var(ω)/2)`, which is exact. The
second treats `κz` as normal, with `Var(κz) = ⟨κ⟩² Var(z) + ⟨z⟩² Var(κ) + Var(κ) Var(z)`, which is
exact only when `κ` or `z` is a point mass.

## Interfaces

```@example gcv
using GCVMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase
MessagePassingRulesBase.nodespec(GCV)
```

| name | meaning | messages its rules take | marginals its rules take |
|---|---|---|---|
| `y` | the output, normal about `x` | univariate normal or `ExponentialLinearQuadratic` (structured) | univariate normal |
| `x` | the mean of `y` | univariate normal or `ExponentialLinearQuadratic` (structured) | univariate normal |
| `z` | the variable controlling the log-variance | none | a normal |
| `κ` | the coupling of `z` to the log-variance | none | anything with a mean and a variance |
| `ω` | the offset of the log-variance | none | anything with a mean and a variance |

No interface of [`GCV`](@ref) has an alias. Under the structured
[factorisation](@extref MessagePassingRulesBase glossary-factorisation) `y` and `x` share one
[cluster](@extref MessagePassingRulesBase glossary-cluster), and so a joint marginal `q(y, x)`, a
bivariate normal.

## Algorithm

[`GCVApproximation`](@ref)`(; method = GaussHermiteCubature(20))` is the node's
[algorithm](@extref MessagePassingRulesBase glossary-algorithm). It carries the
[`GaussHermiteCubature`](@extref MessagePassingRulesApproximations.GaussHermiteCubature) that
computes the moments of the messages towards `z`, `κ` and `ω`. The node declares its default
instance, so a model does not need to name it, and names it only to change the number of points.
The rules are declared for the default instance's type, so another approximation method finds no
rule.

The algorithm declares no dependencies of its own. The rules take what the
[default scheme](@extref MessagePassingRulesBase glossary-default-scheme) delivers: messages from
the target's cluster and marginals of the other clusters.

## Supported rules

```@example gcv
MessagePassingRulesBase.rule_coverage(GCV)
```

Each target of [`GCV`](@ref) has two rules. One is for the structured factorisation
`q(y, x) q(z) q(κ) q(ω)`, and one is for the
[mean field](@extref MessagePassingRulesBase glossary-mean-field), which puts every variable in
a cluster of its own. Under the structured factorisation:

- the rules towards `y` and `x` take the message on the other one and the marginals of `z`, `κ`
  and `ω`;
- the rules towards `z`, `κ` and `ω` take the joint `q(y, x)`;
- the row `q(y, x)` is the marginal rule of the joint.

There are no [belief propagation](@extref MessagePassingRulesBase glossary-belief-propagation)
rules. The [average energy](@extref MessagePassingRulesBase glossary-average-energy) is defined
for both factorisations.

The rules that the package adds to the normal nodes of StandardMessagePassingRules take an
[`ExponentialLinearQuadratic`](@ref) message on `out`, which they reduce to a normal with its
moments:

| node | target | inputs |
|---|---|---|
| `NormalMeanVariance` | `μ` | `m(out)`, and `m(v)` a point mass or `q(v)` |
| `NormalMeanVariance` | `q(out, μ, v)` | `m(out)`, a univariate normal `m(μ)`, a point-mass `m(v)` |
| `NormalMeanVariance` | `q(out, μ)` | `m(out)`, a univariate normal `m(μ)`, `q(v)` |
| `NormalMeanPrecision` | `μ` | `m(out)`, and `m(τ)` a point mass or `q(τ)` |
| `NormalMeanPrecision` | `q(out, μ, τ)` | `m(out)`, a univariate normal `m(μ)`, a point-mass `m(τ)` |
| `NormalMeanPrecision` | `q(out, μ)` | `m(out)`, a univariate normal `m(μ)`, `q(τ)` |

## Example

The message towards `y` from a normal message on `x`, with point-mass marginals that make the noise
variance `exp(1 ⋅ 0 + 0) = 1`:

```@example gcv
@call_message_update_rule(
    node = GCV, target = :y,
    m = (x = NormalMeanVariance(1.0, 2.0),),
    q = (z = PointMass(0.0), κ = PointMass(1.0), ω = PointMass(0.0)),
)
```

The card draws the message on `x` as a solid input and the marginals of `z`, `κ` and `ω` as
dashed ones. The result adds the noise variance to the variance of `x`, `2 + 1 = 3`:

```jldoctest
julia> using MessagePassingRulesBase, GCVMessagePassingRules, ExponentialFamily, BayesBase

julia> result = @call_message_update_rule(
           node = GCV, target = :y,
           m = (x = NormalMeanVariance(1.0, 2.0),),
           q = (z = PointMass(0.0), κ = PointMass(1.0), ω = PointMass(0.0)),
       );

julia> mean_var(getresult(result)) .≈ (1.0, 3.0)
(true, true)
```

The message towards `ω` under the mean field is an [`ExponentialLinearQuadratic`](@ref), whose
moments the cubature of the default algorithm computes. Its coefficient `b` is
`⟨(y - x)²⟩ = (1 - 0)² + 0.5 + 0.5 = 2`:

```jldoctest
julia> using MessagePassingRulesBase, GCVMessagePassingRules, ExponentialFamily, BayesBase

julia> result = @call_message_update_rule(
           node = GCV, target = :ω,
           q = (y = NormalMeanVariance(1.0, 0.5), x = NormalMeanVariance(0.0, 0.5), z = PointMass(0.0), κ = PointMass(1.0)),
       );

julia> message = getresult(result);

julia> message isa ExponentialLinearQuadratic
true

julia> params(message) == (1.0, 2.0, -1.0, 0.0)
true
```

### A step of a hierarchical Gaussian filter

In a two-layer hierarchical Gaussian filter, the upper layer `z` is a random walk,
`zₜ ~ N(zₜ₋₁, 1)`, and it sets the volatility of the lower random walk,
`xₜ ~ GCV(xₜ₋₁, zₜ, κ, ω)`. The update of the upper layer passes through two nodes. First the
[`GCV`](@ref) node sends its message towards `zₜ`, from the marginals of the lower layer:

```@example gcv
towards_z = @call_message_update_rule(
    node = GCV, target = :z,
    q = (y = NormalMeanVariance(1.0, 0.5), x = NormalMeanVariance(0.0, 0.5), κ = PointMass(1.0), ω = PointMass(0.0)),
)
m_z = getresult(towards_z)
mean_var(m_z)
```

The message is an `ExponentialLinearQuadratic`, and its moments come from the cubature. Then the
random walk's [`NormalMeanVariance`](@extref StandardMessagePassingRules NormalMeanVariance)
node carries it back towards `zₜ₋₁`. This is one of the rules the package adds to that node:

```@example gcv
@call_message_update_rule(node = NormalMeanVariance, target = :μ, m = (out = m_z, v = PointMass(1.0)))
```

The rule reduces the message on `out` to a normal with its moments and adds the random walk's
variance, `1`.

## Limitations

- There are no belief-propagation rules: every rule needs the marginals of `z`, `κ` and `ω`, so a
  model must initialise them.
- The node is univariate; the messages on `y` and `x` must be univariate normals or
  [`ExponentialLinearQuadratic`](@ref)s.
- `⟨exp(-κz)⟩` treats `κz` as normal, which is an approximation unless `κ` or `z` is a point mass.
- The approximation method must be a Gauss–Hermite cubature.
- The average energy needs a normal `q(z)`, and a multivariate normal `q(y, x)` or normal `q(y)` and
  `q(x)`.
- The rules declare no [log scale](@extref MessagePassingRulesBase glossary-log-scale).

## API

The node and its algorithm:

```@docs
GCV
GCVApproximation
```

The density of the messages towards `z`, `κ` and `ω`:

```@docs
ExponentialLinearQuadratic
```
