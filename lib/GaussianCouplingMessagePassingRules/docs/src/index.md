# GaussianCouplingMessagePassingRules

The [`GaussianCoupling`](@ref) node couples two scalar variables through a bilinear potential.
It is the edge potential of Gaussian belief propagation (GaBP). Use it to solve a linear system
`A x = b` by message passing, or to find the means of a Gaussian Markov random field given in
information form: every off-diagonal entry of `A` becomes one coupling node.

```@docs
GaussianCouplingMessagePassingRules
```

!!! info "Where these rules run"
    The [ReactiveMP](https://reactivebayes.github.io/ReactiveMP.jl/dev/) engine runs these rules on
    a factor graph, which [RxInfer](https://github.com/ReactiveBayes/RxInfer.jl) builds from a model.
    The examples here call the rules directly.

The package holds one node, so this page is the whole site.

## Overview

A symmetric positive definite matrix `A` and a vector `b` define the Gaussian density
`p(x) ∝ exp(-x'A x / 2 + b'x)`, whose mean is the solution of `A x = b`. Gaussian belief
propagation writes this density as a product of factors on a
[factor graph](@extref MessagePassingRulesBase glossary-factor-graph):

- one self-potential per variable: the prior of `x[i]` is
  `NormalWeightedMeanPrecision(b[i], A[i, i])`;
- one pairwise potential per non-zero `A[i, j]` with `i < j`: a [`GaussianCoupling`](@ref)
  node with coefficient `a = -A[i, j]`.

[Belief propagation](@extref MessagePassingRulesBase glossary-belief-propagation) on this graph
then computes the means and variances of the `x[i]`. On a tree they are exact. On a graph with
cycles, when the iteration converges, the means are exact and the variances approximate.

## Definition

```math
\phi(\mathrm{out}, \mathrm{in}, a) = \exp(\mathrm{out} \cdot a \cdot \mathrm{in})
```

The potential rewards `out` and `in` of the same sign when `a > 0`, and of opposite signs when
`a < 0`. It is not integrable, so the node is not a conditional distribution. Its
[messages](@extref MessagePassingRulesBase glossary-message) are improper normals, with negative
precision, and the model as a whole must be normalisable.

## Interfaces

```@example coupling
using GaussianCouplingMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase
MessagePassingRulesBase.nodespec(GaussianCoupling)
```

| name | aliases | meaning | messages and marginals its rules take |
|:---|:---|:---|:---|
| `out` | none | the first coupled variable | univariate normal messages |
| `in` | none | the second coupled variable | univariate normal messages |
| `a` | none | the coupling coefficient, `-A[i, j]` | a `PointMass` marginal |

[`GaussianCoupling`](@ref) is symmetric in `out` and `in`. The coefficient arrives as a
[point mass](@extref MessagePassingRulesBase glossary-point-mass), a distribution with all its
mass on one value.

## Algorithm

The node uses [`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm), which has
no parameters, so a model names none. It declares no dependencies of its own: its rules take
what the [default scheme](@extref MessagePassingRulesBase glossary-default-scheme) delivers,
messages from the target's [cluster](@extref MessagePassingRulesBase glossary-cluster) and
marginals of the other clusters.

## Supported rules

```@example coupling
MessagePassingRulesBase.rule_coverage(GaussianCoupling)
```

The rules of [`GaussianCoupling`](@ref) assume the
[factorisation](@extref MessagePassingRulesBase glossary-factorisation) `q(out, in) q(a)`, which
puts `out` and `in` in one cluster and `a` in another. Within it, the rules are belief
propagation:

- the message towards `out` reads the message on `in` and the
  [marginal](@extref MessagePassingRulesBase glossary-marginal) `q(a)`, and the message towards
  `in` reads the message on `out` and `q(a)`;
- the joint marginal `q(out, in)` reads both messages and `q(a)`;
- the [average energy](@extref MessagePassingRulesBase glossary-average-energy)
  `⟨-log φ⟩ = -E[a] E[out ⋅ in]` reads `q(out, in)` and `q(a)`.

A constant `a` puts a model in this factorisation without a constraint.

## Example

The message towards `in`, from a message `N(2, 3)` on `out` and the coefficient `a = -0.5`:

```@example coupling
@call_message_update_rule(
    node = GaussianCoupling, target = :in,
    m = (out = NormalMeanVariance(2.0, 3.0),), q = (a = PointMass(-0.5),),
)
```

The card draws the message on `out` and the marginal of `a` as the inputs. The result is
`NormalWeightedMeanPrecision(a ⋅ 2, -a² ⋅ 3)`, with a negative precision. The joint marginal adds
the cross term `-a` to the two incoming precisions:

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

### Solving a two-variable system

For `A = [2 1; 1 3]` and `b = [1, 2]`, the graph has two priors and one coupling with
`a = -A[1, 2] = -1`. It is a tree, so one message gives the exact marginal of `x[2]`:

```@example coupling
A, b = [2.0 1.0; 1.0 3.0], [1.0, 2.0]
prior_1 = NormalWeightedMeanPrecision(b[1], A[1, 1])
prior_2 = NormalWeightedMeanPrecision(b[2], A[2, 2])

message = @call_message_update_rule(
    node = GaussianCoupling, target = :in,
    m = (out = prior_1,), q = (a = PointMass(-A[1, 2]),),
)
marginal_2 = prod(GenericProd(), prior_2, getresult(message))

mean_var(marginal_2), (A \ b)[2], inv(A)[2, 2]
```

The marginal's mean is the second entry of the solution `A \ b`, and its variance is the second
diagonal entry of `A⁻¹`.

## Limitations

- Scalar variables only: the rules take univariate normal messages, and no multivariate version
  exists.
- The coefficient `a` must be a `PointMass`; a random coefficient has no rules.
- Only the structured factorisation `q(out, in) q(a)`: there are no
  [mean-field](@extref MessagePassingRulesBase glossary-mean-field) rules and no mean-field
  average energy.
- The messages are improper by design, and the joint `q(out, in)` is proper only when
  `w_out ⋅ w_in > a²`, `w` being the incoming precisions.
- The rules declare no [log scale](@extref MessagePassingRulesBase glossary-log-scale).
- On graphs with cycles the variances are approximations of `diag(A⁻¹)`, not the exact marginal
  variances. The means are exact when GaBP converges, for example when `A` is strictly
  diagonally dominant.

## API

The node, with its interfaces, factorisation, rules and the conditions under which GaBP is exact:

```@docs
GaussianCoupling
```
