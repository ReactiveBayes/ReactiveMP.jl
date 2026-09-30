# Normal distributions

A normal node ties a variable `out` to a mean and a spread: a variance, a precision, a
covariance matrix or a precision matrix. The page covers the univariate and multivariate normals
in each of their parametrisations, and the half-normal.

With the spread known, the rules towards `out` and the mean are exact
[belief propagation](@extref MessagePassingRulesBase glossary-belief-propagation): a normal
[message](@extref MessagePassingRulesBase glossary-message) on the mean, convolved with the noise,
gives a normal message on `out`, and the same holds the other way. With the spread unknown,
belief propagation towards it gives no closed-form message for `NormalMeanVariance`, and the
other normals have no such rule at all. The rules towards the spread are therefore
[variational](@extref MessagePassingRulesBase glossary-vmp): they take
[marginals](@extref MessagePassingRulesBase glossary-marginal) and return a Gamma, a Wishart or
their inverses. In practice, you choose a
[factorisation](@extref MessagePassingRulesBase glossary-factorisation) that puts an unknown
spread in a [cluster](@extref MessagePassingRulesBase glossary-cluster) of its own, such as
`q(out, μ) q(τ)` or the [mean field](@extref MessagePassingRulesBase glossary-mean-field)
`q(out) q(μ) q(τ)`.

The nodes are ExponentialFamily's types, declared as nodes here. They all run under
[`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm), so a model never names an
algorithm for them.

In the interface tables below, a rule takes *messages* from the interfaces in its target's own
cluster, and it takes *marginals* from the interfaces in other clusters. "Normal" means any
univariate or multivariate normal of ExponentialFamily in any parametrisation. "Any" means a
marginal with the expectations the rule needs.

```@setup normal
using MessagePassingRulesBase, StandardMessagePassingRules, ExponentialFamily, Distributions, BayesBase
```

## Example

The mean-field message towards the mean of a `NormalMeanPrecision` node takes the marginals of
`out` and of the precision. It is a normal around `E[out]` with precision `E[τ]`:

```jldoctest normal
julia> using StandardMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase

julia> message = getresult(@call_message_update_rule(
           node = NormalMeanPrecision, target = :μ,
           q = (out = PointMass(2.0), τ = GammaShapeRate(2.0, 1.0)),
       ));

julia> mean(message) ≈ 2.0 && precision(message) ≈ 2.0
true
```

## NormalMeanVariance

```math
p(\mathrm{out} \mid μ, v) = \frac{1}{\sqrt{2π v}} \exp\left(-\frac{(\mathrm{out} - μ)^2}{2v}\right)
```

```@example normal
MessagePassingRulesBase.nodespec(NormalMeanVariance)
```

| interface | aliases | meaning | messages and marginals its rules take |
|---|---|---|---|
| `out` | | the variable | a univariate normal or a `PointMass` |
| `μ` | `mean` | the mean | a univariate normal or a `PointMass` |
| `v` | `var` | the variance | a `PointMass` message, or any marginal with `E[1/v]` |

```@example normal
MessagePassingRulesBase.rule_coverage(NormalMeanVariance)
```

The belief propagation message towards `out`, from a normal message on the mean and a known
variance, adds the variances:

```@example normal
@call_message_update_rule(
    node = NormalMeanVariance, target = :out,
    m = (μ = NormalMeanVariance(1.0, 1.0), v = PointMass(2.0)),
)
```

The variational message takes the marginals of `μ` and `v`. It keeps the mean of `q(μ)`, ignores
its variance, and uses `1/E[1/v] = 0.5` of the Gamma marginal as the variance:

```@example normal
@call_message_update_rule(
    node = NormalMeanVariance, target = :out,
    q = (μ = NormalMeanVariance(1.0, 1.0), v = GammaShapeRate(3.0, 4.0)),
)
```

Towards `out` and `μ`, the node has belief propagation, variational and structured rules. Towards
`v`, belief propagation from normal or known `out` and `μ` gives a log-density on the half line, a
`ContinuousUnivariateLogPdf`, which has no closed-form family. The variational rule towards `v`
gives an unchecked `GammaInverse` with shape `-1/2`, which is a likelihood rather than a
distribution. The average energy covers the mean field and `q(out, μ) q(v)`.

## NormalMeanPrecision

```math
p(\mathrm{out} \mid μ, τ) = \sqrt{\frac{τ}{2π}} \exp\left(-\frac{τ (\mathrm{out} - μ)^2}{2}\right)
```

```@example normal
MessagePassingRulesBase.nodespec(NormalMeanPrecision)
```

| interface | aliases | meaning | messages and marginals its rules take |
|---|---|---|---|
| `out` | | the variable | a univariate normal or a `PointMass` |
| `μ` | `mean` | the mean | a univariate normal or a `PointMass` |
| `τ` | `invcov`, `precision` | the precision | a `PointMass` message, or any marginal with `E[τ]` |

```@example normal
MessagePassingRulesBase.rule_coverage(NormalMeanPrecision)
```

With a known precision, the message towards `out` stays in precision form. The precisions combine
as `1 / (1/1 + 1/2) = 2/3`:

```@example normal
@call_message_update_rule(
    node = NormalMeanPrecision, target = :out,
    m = (μ = NormalMeanPrecision(1.0, 1.0), τ = PointMass(2.0)),
)
```

The variational message towards `τ` takes the marginals of `out` and `μ`, and it is a `Gamma`:

```@example normal
@call_message_update_rule(
    node = NormalMeanPrecision, target = :τ,
    q = (out = PointMass(3.0), μ = NormalMeanPrecision(1.0, 1.0)),
)
```

The rules towards `τ` take `q(out) q(μ)` or `q(out, μ)`. There is **no belief propagation rule
towards `τ`**, so an unknown precision needs a factorisation that separates it from `out` and
`μ`. Asking for one throws a
[`RuleNotFoundError`](@extref MessagePassingRulesBase.RuleNotFoundError), which lists the rules
that do exist:

```@example normal
try
    @call_message_update_rule(
        node = NormalMeanPrecision, target = :τ,
        m = (out = PointMass(3.0), μ = NormalMeanPrecision(1.0, 1.0)),
    )
catch err
    showerror(stdout, err)
end
```

## MvNormalMeanCovariance

```math
p(\mathrm{out} \mid μ, Σ) = |2π Σ|^{-1/2}
\exp\left(-\tfrac{1}{2} (\mathrm{out} - μ)^\top Σ^{-1} (\mathrm{out} - μ)\right)
```

```@example normal
MessagePassingRulesBase.nodespec(MvNormalMeanCovariance)
```

| interface | aliases | meaning | messages and marginals its rules take |
|---|---|---|---|
| `out` | | the variable | a multivariate normal or a `PointMass` |
| `μ` | `mean` | the mean | a multivariate normal or a `PointMass` |
| `Σ` | `cov` | the covariance | a `PointMass` message, or any marginal with `E[Σ⁻¹]` |

```@example normal
MessagePassingRulesBase.rule_coverage(MvNormalMeanCovariance)
```

As in the univariate case, the belief propagation message towards `out` adds the covariances:

```@example normal
@call_message_update_rule(
    node = MvNormalMeanCovariance, target = :out,
    m = (μ = MvNormalMeanCovariance([1.0, 2.0], diageye(2)), Σ = PointMass([2.0 0.0; 0.0 2.0])),
)
```

The rules towards `Σ` are variational only, and they give an inverse-Wishart likelihood. There is
no belief propagation rule towards `Σ`.

## MvNormalMeanPrecision

```math
p(\mathrm{out} \mid μ, Λ) = \left|\frac{Λ}{2π}\right|^{1/2}
\exp\left(-\tfrac{1}{2} (\mathrm{out} - μ)^\top Λ (\mathrm{out} - μ)\right)
```

| interface | aliases | meaning | messages and marginals its rules take |
|---|---|---|---|
| `out` | | the variable | a multivariate normal or a `PointMass` |
| `μ` | `mean` | the mean | a multivariate normal or a `PointMass` |
| `Λ` | `invcov`, `precision` | the precision matrix | a `PointMass` message, or any marginal with `E[Λ]`; a `Wishart` marginal has rules of its own |

```@example normal
MessagePassingRulesBase.rule_coverage(MvNormalMeanPrecision)
```

The rules towards `Λ` are variational only, and they give a Wishart likelihood. They apply the
context's [`matrix_correction`](@extref MessagePassingRulesBase.matrix_correction) to its scale,
and none by default. There is no belief propagation rule towards `Λ`.

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

**Limitations.** The node has the message towards `out` and the joint marginal with known
parameters only. There is **no rule towards `ξ` or `Λ`**: they must be known, or have their
marginals from elsewhere.

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

Every message rule takes `γ` as a marginal, so `γ` is always in a cluster of its own. The rules
towards `γ` are variational, and they give a `GammaShapeRate`.

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

As for `MvNormalMeanScalePrecision`, every message rule takes `γ` and `G` as marginals. The rules
towards them are variational.

## HalfNormal

```math
p(\mathrm{out} \mid v) = \frac{2}{\sqrt{2π v}} \exp\left(-\frac{\mathrm{out}^2}{2v}\right),
\quad \mathrm{out} \ge 0
```

| interface | aliases | meaning | messages and marginals its rules take |
|---|---|---|---|
| `out` | | the variable | any marginal (average energy) |
| `v` | `var`, `σ²` | the variance | a `PointMass` marginal (message towards `out`), any marginal (average energy) |

```@example normal
MessagePassingRulesBase.rule_coverage(HalfNormal)
```

**Limitations.** [`HalfNormal`](@ref) has one message, a truncated normal towards `out` from a
known variance. There is **no rule towards `v`**, and no marginal rule.

```@docs
HalfNormal
```
