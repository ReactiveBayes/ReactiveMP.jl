# GaussianCouplingMessagePassingRules

The `GaussianCoupling` node couples two scalar variables through a bilinear potential, and is
the edge potential of Gaussian belief propagation (GaBP). Use it to solve a linear system
`A x = b`, or to find the means of a Gaussian Markov random field in information form, by
message passing: every off-diagonal entry of `A` becomes one coupling node.

```@docs
GaussianCouplingMessagePassingRules
```

The package holds one node, so this page is the whole site.

## Overview

Gaussian belief propagation writes the density `p(x) ∝ exp(-x'A x / 2 + b'x)` as a product of
self-potentials, one per variable, and pairwise potentials, one per non-zero `A[i, j]` with
`i < j`. The self-potential of `x[i]` is the prior `NormalWeightedMeanPrecision(b[i], A[i, i])`;
the pairwise potential is a `GaussianCoupling` node with coefficient `a = -A[i, j]`. On a tree
the posterior means and variances are exact; on a graph with cycles, when the iteration
converges, the means are exact and the variances approximate.

## Definition

```math
\phi(\mathrm{out}, \mathrm{in}, a) = \exp(\mathrm{out} \cdot a \cdot \mathrm{in})
```

The potential is not integrable, so the node is not a conditional distribution. Its messages
are improper normals with negative precision, and the model as a whole must be normalisable.

## Interfaces

| name | aliases | meaning | messages and marginals its rules take |
|:---|:---|:---|:---|
| `out` | none | the first coupled variable | univariate normal messages |
| `in` | none | the second coupled variable | univariate normal messages |
| `a` | none | the coupling coefficient, `-A[i, j]` | a `PointMass` marginal |

The node is symmetric in `out` and `in`.

## Algorithm

The node uses [`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm), which has
no parameters; a model names none.

## Supported rules

```@example
using MessagePassingRulesBase, GaussianCouplingMessagePassingRules # hide
MessagePassingRulesBase.rule_coverage(GaussianCoupling)
```

The rules are belief propagation within the structured factorisation `q(out, in) q(a)`: the
messages towards `out` and `in` read the message on the other variable and `q(a)`, the joint
marginal `q(out, in)` reads both messages and `q(a)`, and the average energy
`⟨-log φ⟩ = -E[a] E[out ⋅ in]` reads `q(out, in)` and `q(a)`. A constant `a` puts a model in this
factorisation without a constraint.

## Example

The message towards `in`, from a message `N(2, 3)` on `out` and the coefficient `a = -0.5`, is
`NormalWeightedMeanPrecision(a ⋅ 2, -a² ⋅ 3)`, with negative precision:

```jldoctest
julia> using GaussianCouplingMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase

julia> result = @call_message_update_rule(
           node = GaussianCoupling, target = :in,
           m = (out = NormalMeanVariance(2.0, 3.0),), q = (a = PointMass(-0.5),),
       );

julia> getresult(result) ≈ NormalWeightedMeanPrecision(-1.0, -0.75)
true

julia> joint = @call_marginal_update_rule(
           node = GaussianCoupling, target = (:out, :in),
           m = (out = NormalMeanVariance(1.0, 0.5), in = NormalMeanVariance(-2.0, 0.25)),
           q = (a = PointMass(-1.5),),
       );

julia> getresult(joint) ≈ MvNormalWeightedMeanPrecision([2.0, -8.0], [2.0 1.5; 1.5 4.0])
true
```

In an RxInfer model, solving `A x = b` for a symmetric `A`:

```julia
@model function gabp(A, b)
    n = length(b)
    for i in 1:n
        x[i] ~ NormalWeightedMeanPrecision(b[i], A[i, i])
    end
    for i in 1:n, j in (i + 1):n
        if !iszero(A[i, j])
            x[j] ~ GaussianCoupling(x[i], -A[i, j])
        end
    end
end
```

## Limitations

- Scalar variables only: the rules take univariate normal messages, and no multivariate version
  exists.
- The coefficient `a` must be a `PointMass`; a random coefficient has no rules.
- Only the structured factorisation `q(out, in) q(a)`: there are no mean-field rules and no
  mean-field average energy.
- The messages are improper by design, and the joint `q(out, in)` is proper only when
  `w_out ⋅ w_in > a²`, `w` being the incoming precisions.
- The rules declare no log scale.
- On graphs with cycles the variances are approximations of `diag(A⁻¹)`, not the exact marginal
  variances; the means are exact when GaBP converges, for example when `A` is strictly
  diagonally dominant.

## API

The node, with its interfaces, factorisation, rules and the conditions under which GaBP is exact:

```@docs
GaussianCoupling
```
