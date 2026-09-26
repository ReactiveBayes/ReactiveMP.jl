# Matrix and joint distributions

The distributions over matrices, and the joint normal–Gamma and normal–Wishart priors. They
are the types of ExponentialFamily and Distributions, declared as nodes here, and all run under
[`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm). Most of them are priors
over precision or covariance matrices, with the message towards `out` only. The rules work with
ExponentialFamily's fast forms, `WishartFast` and `InverseWishartFast`, which store what the
updates need; a marginal is formed as Distributions' `Wishart` and `InverseWishart`.

```@setup matrix
using MessagePassingRulesBase, StandardMessagePassingRules, ExponentialFamily, Distributions
```

## Example

A known Wishart prior sends its own distribution towards `out`:

```jldoctest matrix
julia> using StandardMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase

julia> message = getresult(@call_message_update_rule(
           node = Wishart, target = :out,
           m = (ν = PointMass(3.0), S = PointMass(diageye(2))),
       ));

julia> mean(message) ≈ 3 * diageye(2)
true
```

```julia
@model function covariance(y)
    Σ ~ InverseWishart(4.0, diageye(2))
    y .~ MvNormalMeanCovariance(zeros(2), Σ)
end
```

## Wishart

```math
p(\mathrm{out} \mid ν, S) = \frac{|\mathrm{out}|^{(ν - d - 1)/2} e^{-\mathrm{tr}(S^{-1} \mathrm{out})/2}}{2^{ν d / 2} |S|^{ν / 2} Γ_d(ν / 2)}
```

| interface | aliases | meaning | messages and marginals its rules take |
|---|---|---|---|
| `out` | | the positive-definite matrix | a Wishart message (marginal rule), any marginal (average energy) |
| `ν` | `df` | the degrees of freedom | a `PointMass` message, or any marginal with `E[ν]` |
| `S` | `scale` | the scale matrix | a `PointMass` message, or any marginal with `E[S⁻¹]` |

```@example matrix
MessagePassingRulesBase.rule_coverage(Wishart)
```

**Limitations.** No rules towards `ν` or `S`. The average energy needs a known `ν`.

## InverseWishart

```math
p(\mathrm{out} \mid ν, S) = \frac{|S|^{ν/2} |\mathrm{out}|^{-(ν + d + 1)/2} e^{-\mathrm{tr}(S \mathrm{out}^{-1})/2}}{2^{ν d / 2} Γ_d(ν / 2)}
```

| interface | aliases | meaning | messages and marginals its rules take |
|---|---|---|---|
| `out` | | the positive-definite matrix | an inverse-Wishart message (marginal rule), any marginal (average energy) |
| `ν` | `df` | the degrees of freedom | a `PointMass` message, or any marginal with `E[ν]` |
| `S` | `scale`, `Ψ` | the scale matrix | a `PointMass` message, or any marginal with `E[S]` |

```@example matrix
MessagePassingRulesBase.rule_coverage(InverseWishart)
```

**Limitations.** No rules towards `ν` or `S`. The average energy needs a known `ν`.

## MatrixNormal

An `n`×`p` matrix normal with row covariance `U` and column covariance `V`:

```math
p(\mathrm{out} \mid M, U, V) = \frac{\exp\left(-\tfrac{1}{2} \mathrm{tr}\left[V^{-1} (\mathrm{out} - M)^\top U^{-1} (\mathrm{out} - M)\right]\right)}{(2π)^{np/2} |V|^{n/2} |U|^{p/2}}
```

| interface | aliases | meaning | messages and marginals its rules take |
|---|---|---|---|
| `out` | | the matrix | a `PointMass` or a `MatrixNormal` |
| `M` | `mean` | the mean matrix | a `PointMass` or a `MatrixNormal` |
| `U` | `rowcov` | the row covariance | a `PointMass` or an inverse-Wishart |
| `V` | `colcov` | the column covariance | a `PointMass` or an inverse-Wishart |

```@example matrix
MessagePassingRulesBase.rule_coverage(MatrixNormal)
```

Towards `out` and `M`, belief propagation takes at most one of `M` (or `out`), `U` and `V`
uncertain; the variational rules take any combination of these types. Towards `U` and `V` the
message is an inverse-Wishart, by belief propagation only from known `out`, `M` and the other
covariance. The joint marginal is defined for known inputs only.

## MatrixNormalWishart

The joint `out = (X, Y)` of a matrix normal and a Wishart:
`X | Y ~ MatrixNormal(M, U, Y⁻¹)` and `Y ~ Wishart(ν, V)`.

| interface | aliases | meaning | messages and marginals its rules take |
|---|---|---|---|
| `out` | | the pair `(X, Y)` | a `MatrixNormalWishart` marginal (average energy) |
| `M` | `mean` | the mean matrix | a `PointMass` marginal |
| `U` | `rowcov` | the row covariance | a `PointMass` marginal |
| `V` | `scale` | the Wishart scale | a `PointMass` marginal |
| `ν` | `dof` | the Wishart degrees of freedom | a `PointMass` marginal |

```@example matrix
MessagePassingRulesBase.rule_coverage(MatrixNormalWishart)
```

**Limitations.** A prior with known parameters: only the message towards `out`, from point-mass
marginals, and no rules towards the parameters.

## MvNormalGamma

The joint `out = (θ, γ)` of a normal and a Gamma precision scale:
`θ | γ ~ N(μ, (γ Λ)⁻¹)` and `γ ~ GammaShapeRate(α, β)`.

| interface | aliases | meaning | messages and marginals its rules take |
|---|---|---|---|
| `out` | | the pair `(θ, γ)` | an `MvNormalGamma` marginal (average energy) |
| `μ` | | the mean | a `PointMass` message, or any marginal with its covariance |
| `Λ` | | the precision matrix | a `PointMass` message, or any marginal with `E[Λ]` |
| `α` | | the Gamma shape | a `PointMass` message, or any marginal with `E[α]` |
| `β` | | the Gamma rate | a `PointMass` message, or any marginal with `E[β]` |

```@example matrix
MessagePassingRulesBase.rule_coverage(MvNormalGamma)
```

**Limitations.** No rules towards the parameters. The average energy needs them all known.

## MvNormalWishart

The joint `out = (x, Λ)` of a normal and a Wishart precision:
`x | Λ ~ N(μ, (λ Λ)⁻¹)` and `Λ ~ Wishart(ν, W)`. It is ExponentialFamily's
`MvNormalWishart(μ, Ψ, κ, ν)`, with `W` for `Ψ` and `λ` for `κ`.

| interface | aliases | meaning | messages and marginals its rules take |
|---|---|---|---|
| `out` | | the pair `(x, Λ)` | none |
| `μ` | `mean` | the mean | a `PointMass` marginal |
| `W` | `scale` | the Wishart scale | a `PointMass` marginal |
| `λ` | | the precision scale | a `PointMass` marginal |
| `ν` | | the degrees of freedom | a `PointMass` marginal |

```@example matrix
MessagePassingRulesBase.rule_coverage(MvNormalWishart)
```

**Limitations.** Only the message towards `out`, from known parameters, and **no average
energy**: the free energy of a model with this node is an error. The alias `scale` names `W`,
although ExponentialFamily's `scale` of an `MvNormalWishart` is `κ`, the `λ` interface here.
