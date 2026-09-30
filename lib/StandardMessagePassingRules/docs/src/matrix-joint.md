# Matrix and joint distributions

These nodes model random matrices and joint priors. The Wishart and inverse-Wishart nodes are
distributions over positive-definite matrices, such as a precision or a covariance. The matrix
normal is a normal over a matrix, with one covariance for its rows and one for its columns. The
joint nodes put a prior on a vector and its precision at once: a normal with a Gamma precision
scale, or a normal with a Wishart precision.

Most of these nodes serve as priors. With known parameters, the
[message](@extref MessagePassingRulesBase glossary-message) towards `out` is the node's own
distribution, which is exact
[belief propagation](@extref MessagePassingRulesBase glossary-belief-propagation). They have no
rules towards their parameters, so a model gives the parameters as constants. `MatrixNormal` is
the exception. It has belief propagation rules towards every interface, and
[variational](@extref MessagePassingRulesBase glossary-vmp) rules, which take
[marginals](@extref MessagePassingRulesBase glossary-marginal), towards every interface too.

The nodes are the types of ExponentialFamily and Distributions, declared as nodes here, and all
run under [`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm). The rules work
with ExponentialFamily's fast forms, `WishartFast` and `InverseWishartFast`, which store what the
updates need. A marginal of these types is formed as Distributions' `Wishart` or `InverseWishart`.

```@setup matrix
using MessagePassingRulesBase, StandardMessagePassingRules, ExponentialFamily, Distributions, BayesBase
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

**Limitations.** No rules towards `ν` or `S`. The
[average energy](@extref MessagePassingRulesBase glossary-average-energy) needs a known `ν`.

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
MessagePassingRulesBase.nodespec(MatrixNormal)
```

```@example matrix
MessagePassingRulesBase.rule_coverage(MatrixNormal)
```

Towards `out` and `M`, the belief propagation rules take at most one uncertain input among `M`
(or `out`), `U` and `V`. The variational rules take any combination of these types. Towards `U`
and `V`, the message is an inverse-Wishart. The belief propagation rule needs `out`, `M` and the
other covariance known. The variational rule takes a `PointMass` or `MatrixNormal` marginal of
`out` and `M`, and any marginal of the other covariance. The joint marginal is defined for known
inputs only.

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

**Limitations.** `MatrixNormalWishart` is a prior with known parameters. It has the message
towards `out` only, from [point-mass](@extref MessagePassingRulesBase glossary-point-mass)
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

**Limitations.** The node has the message towards `out` only, from known parameters. It has **no
[average energy](@extref MessagePassingRulesBase glossary-average-energy)**, so the free energy
of a model with this node is an error. The alias `scale` names `W`, although ExponentialFamily's
`scale` of an `MvNormalWishart` is `κ`, the `λ` interface here.
