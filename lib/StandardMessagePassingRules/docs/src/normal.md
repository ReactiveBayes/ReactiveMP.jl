# Normal distributions

The univariate and multivariate normal nodes, in each of their parametrisations, and the
half-normal. They are ExponentialFamily's types, declared as nodes here, and all run under
[`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm): a model never names an
algorithm for them. Their rules cover belief propagation with known parameters, mean-field
variational message passing and the structured factorisation that keeps `out` and the mean
together, `q(out, μ) q(parameter)`.

In the tables below, *messages* are what a rule takes from an interface in its own cluster and
*marginals* what it takes from one in another cluster. "Normal" means any univariate or
multivariate normal of ExponentialFamily in any parametrisation, and "any" a marginal with the
expectations the rule needs.

```@setup normal
using MessagePassingRulesBase, StandardMessagePassingRules, ExponentialFamily, Distributions
```

## Example

The mean-field message towards the mean of a `NormalMeanPrecision` node, from the marginals of
`out` and of the precision, is a normal around `E[out]` with precision `E[τ]`:

```jldoctest normal
julia> using StandardMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase

julia> message = getresult(@call_message_update_rule(
           node = NormalMeanPrecision, target = :μ,
           q = (out = PointMass(2.0), τ = GammaShapeRate(2.0, 1.0)),
       ));

julia> mean(message) ≈ 2.0 && precision(message) ≈ 2.0
true
```

```julia
@model function linear_gaussian(y)
    x ~ MvNormalMeanCovariance(zeros(2), diageye(2))
    Λ ~ Wishart(3, diageye(2))
    y ~ MvNormalMeanPrecision(x, Λ)
end
```

## NormalMeanVariance

```math
p(\mathrm{out} \mid μ, v) = \frac{1}{\sqrt{2π v}} \exp\left(-\frac{(\mathrm{out} - μ)^2}{2v}\right)
```

| interface | aliases | meaning | messages and marginals its rules take |
|---|---|---|---|
| `out` | | the variable | a univariate normal or a `PointMass` |
| `μ` | `mean` | the mean | a univariate normal or a `PointMass` |
| `v` | `var` | the variance | a `PointMass` message, or any marginal with `E[1/v]` |

```@example normal
MessagePassingRulesBase.rule_coverage(NormalMeanVariance)
```

Towards `out` and `μ` there are belief propagation, variational and structured rules. Towards
`v`, belief propagation from normal or known `out` and `μ` gives a log-density on the half line,
a `ContinuousUnivariateLogPdf`, with no closed-form family; the variational rule gives an
unchecked `GammaInverse` with shape `-1/2`, a likelihood rather than a distribution. The average
energy covers mean field and `q(out, μ) q(v)`.

## NormalMeanPrecision

```math
p(\mathrm{out} \mid μ, τ) = \sqrt{\frac{τ}{2π}} \exp\left(-\frac{τ (\mathrm{out} - μ)^2}{2}\right)
```

| interface | aliases | meaning | messages and marginals its rules take |
|---|---|---|---|
| `out` | | the variable | a univariate normal or a `PointMass` |
| `μ` | `mean` | the mean | a univariate normal or a `PointMass` |
| `τ` | `invcov`, `precision` | the precision | a `PointMass` message, or any marginal with `E[τ]` |

```@example normal
MessagePassingRulesBase.rule_coverage(NormalMeanPrecision)
```

The rules towards `τ` are variational only, from `q(out) q(μ)` or `q(out, μ)`, and give a
`Gamma`; there is **no belief propagation rule towards `τ`**, so an unknown precision needs a
factorisation that separates it from `out` and `μ`.

## MvNormalMeanCovariance

```math
p(\mathrm{out} \mid μ, Σ) = |2π Σ|^{-1/2}
\exp\left(-\tfrac{1}{2} (\mathrm{out} - μ)^\top Σ^{-1} (\mathrm{out} - μ)\right)
```

| interface | aliases | meaning | messages and marginals its rules take |
|---|---|---|---|
| `out` | | the variable | a multivariate normal or a `PointMass` |
| `μ` | `mean` | the mean | a multivariate normal or a `PointMass` |
| `Σ` | `cov` | the covariance | a `PointMass` message, or any marginal with `E[Σ⁻¹]` |

```@example normal
MessagePassingRulesBase.rule_coverage(MvNormalMeanCovariance)
```

Towards `Σ` the rules are variational only and give an inverse-Wishart likelihood; there is no
belief propagation rule towards `Σ`.

## MvNormalMeanPrecision

```math
p(\mathrm{out} \mid μ, Λ) = \left|\frac{Λ}{2π}\right|^{1/2}
\exp\left(-\tfrac{1}{2} (\mathrm{out} - μ)^\top Λ (\mathrm{out} - μ)\right)
```

| interface | aliases | meaning | messages and marginals its rules take |
|---|---|---|---|
| `out` | | the variable | a multivariate normal or a `PointMass` |
| `μ` | `mean` | the mean | a multivariate normal or a `PointMass` |
| `Λ` | `invcov`, `precision` | the precision matrix | a `PointMass` message, or any marginal with `E[Λ]`; a `Wishart` has its own rules |

```@example normal
MessagePassingRulesBase.rule_coverage(MvNormalMeanPrecision)
```

Towards `Λ` the rules are variational only and give a Wishart likelihood; they apply the context's `matrix_correction` to its scale, and none by default.
There is no belief propagation rule towards `Λ`.

## MvNormalWeightedMeanPrecision

```math
p(\mathrm{out} \mid ξ, Λ) = \mathcal{N}(\mathrm{out} \mid Λ^{-1} ξ, Λ^{-1})
```

| interface | aliases | meaning | messages and marginals its rules take |
|---|---|---|---|
| `out` | | the variable | a multivariate normal |
| `ξ` | `xi`, `weightedmean` | the weighted mean `Λ μ` | a `PointMass` message, or any marginal |
| `Λ` | `invcov`, `precision` | the precision matrix | a `PointMass` message, or any marginal |

```@example normal
MessagePassingRulesBase.rule_coverage(MvNormalWeightedMeanPrecision)
```

Only the message towards `out`, and the joint marginal with known parameters: **no rule towards
`ξ` or `Λ`**, which must be known or have their marginals from elsewhere.

## MvNormalMeanScalePrecision

```math
p(\mathrm{out} \mid μ, γ) = \mathcal{N}(\mathrm{out} \mid μ, (γ I)^{-1})
```

| interface | aliases | meaning | messages and marginals its rules take |
|---|---|---|---|
| `out` | | the variable | a multivariate normal message, or any marginal |
| `μ` | `mean` | the mean | a multivariate normal message, or any marginal |
| `γ` | `precision` | the scalar precision | any marginal with `E[γ]` |

```@example normal
MessagePassingRulesBase.rule_coverage(MvNormalMeanScalePrecision)
```

Every message rule takes `γ` as a marginal, so `γ` is always in a cluster of its own; towards
`γ` the rules are variational and give a `GammaShapeRate`.

## MvNormalMeanScaleMatrixPrecision

```math
p(\mathrm{out} \mid μ, γ, G) = \mathcal{N}(\mathrm{out} \mid μ, (γ G)^{-1})
```

| interface | aliases | meaning | messages and marginals its rules take |
|---|---|---|---|
| `out` | | the variable | a multivariate normal message, or any marginal |
| `μ` | `mean` | the mean | a multivariate normal message, or any marginal |
| `γ` | `scale` | the scalar factor of the precision | any marginal with `E[γ]` |
| `G` | `matrix` | the matrix factor of the precision | any marginal with `E[G]` |

```@example normal
MessagePassingRulesBase.rule_coverage(MvNormalMeanScaleMatrixPrecision)
```

As for `MvNormalMeanScalePrecision`, every message rule takes `γ` and `G` as marginals, and the
rules towards them are variational.

## HalfNormal

```math
p(\mathrm{out} \mid v) = \frac{2}{\sqrt{2π v}} \exp\left(-\frac{\mathrm{out}^2}{2v}\right),
\quad \mathrm{out} \ge 0
```

| interface | aliases | meaning | messages and marginals its rules take |
|---|---|---|---|
| `out` | | the variable | none: nothing is sent from `out` |
| `v` | `var`, `σ²` | the variance | a `PointMass` marginal |

```@example normal
MessagePassingRulesBase.rule_coverage(HalfNormal)
```

**Limitations.** The only message is towards `out`, a truncated normal, from a known variance.
There is **no rule towards `v`**, and no marginal rule.

```@docs
HalfNormal
```
