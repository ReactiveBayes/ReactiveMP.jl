# GCVMessagePassingRules

The Gaussian controlled variance node, `GCV`, and its rules for reactive message passing: a
normal whose log-variance is linear in other variables of the model. Use it for a variance that is
itself inferred and changes over time, as in hierarchical Gaussian filters, where the state of one
layer sets the volatility of the layer below.

```@docs
GCVMessagePassingRules
```

## Overview

A `GCV` node relates an output `y` to its mean `x` with a variance `exp(κz + ω)`: `z` is a latent
variable, often a random walk of its own, `κ` scales its effect on the log-variance and `ω` is an
offset. The rules are variational, and every one of them reads the marginals of `z`, `κ` and `ω`.
The messages towards `z`, `κ` and `ω` are [`ExponentialLinearQuadratic`](@ref) densities, whose
moments a Gauss–Hermite cubature computes; the package also teaches `NormalMeanVariance` and
`NormalMeanPrecision` to take such a message on their `out`, so that `z` or `ω` can be the output of
a normal node.

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

| name | meaning | messages its rules take | marginals its rules take |
|---|---|---|---|
| `y` | the output, normal about `x` | univariate normal or `ExponentialLinearQuadratic` (structured) | univariate normal |
| `x` | the mean of `y` | univariate normal or `ExponentialLinearQuadratic` (structured) | univariate normal |
| `z` | the variable controlling the log-variance | none | a normal |
| `κ` | the coupling of `z` to the log-variance | none | anything with a mean and a variance |
| `ω` | the offset of the log-variance | none | anything with a mean and a variance |

No interface has an alias. Under the structured factorisation `y` and `x` share a joint marginal,
`q(y, x)`, a bivariate normal.

## Algorithm

```julia
GCVApproximation(; method = GaussHermiteCubature(20))
```

[`GCVApproximation`](@ref) carries the
[`GaussHermiteCubature`](@extref MessagePassingRulesApproximations.GaussHermiteCubature) that
computes the moments of the messages towards `z`, `κ` and `ω`. The node declares its default
instance, so a model does not need to name it, and names it only to change the number of points.
The rules are declared for the default instance's type, so another approximation method finds no
rule.

## Supported rules

```@example
using MessagePassingRulesBase, GCVMessagePassingRules # hide
MessagePassingRulesBase.rule_coverage(GCV)
```

Each target has two rules: one for the structured factorisation `q(y, x) q(z) q(κ) q(ω)` and one
for the mean field. Under the structured one, the rules towards `y` and `x` take the message on the
other and the marginals of `z`, `κ` and `ω`, and the rules towards `z`, `κ` and `ω` take `q(y, x)`;
the row `q(y, x)` is the marginal rule of the joint. There are no belief-propagation rules. The
average energy is defined for both factorisations.

The rules that the package adds to the normal nodes of StandardMessagePassingRules take an
`ExponentialLinearQuadratic` message on `out`, which they reduce to a normal of its moments:

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

The message towards `ω` under the mean field is an `ExponentialLinearQuadratic`, whose moments the
cubature of the default algorithm computes:

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

In a model, a two-layer hierarchical Gaussian filter: `z` is a random walk that sets the volatility
of the random walk `x`, under the structured factorisation.

```julia
@model function hgf(y, κ, ω, z_prior, x_prior)
    z_prev ~ z_prior
    x_prev ~ x_prior
    for t in eachindex(y)
        z[t] ~ Normal(mean = z_prev, variance = 1.0)
        x[t] ~ GCV(x_prev, z[t], κ, ω)
        y[t] ~ Normal(mean = x[t], variance = 1.0)
        z_prev, x_prev = z[t], x[t]
    end
end

constraints = @constraints begin
    q(x_prev, x, z) = q(x_prev, x)q(z)
end

# Only to change the cubature's number of points:
algorithm = @algorithm begin
    GCV() -> GCVApproximation(method = GaussHermiteCubature(32))
end
```

## Limitations

- There are no belief-propagation rules: every rule needs the marginals of `z`, `κ` and `ω`, so a
  model must initialise them.
- The node is univariate; the messages on `y` and `x` must be univariate normals or
  `ExponentialLinearQuadratic`s.
- `⟨exp(-κz)⟩` treats `κz` as normal, which is an approximation unless `κ` or `z` is a point mass.
- The approximation method must be a Gauss–Hermite cubature.
- The average energy needs a normal `q(z)`, and a multivariate normal `q(y, x)` or normal `q(y)` and
  `q(x)`.
- The rules declare no log scale.

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
