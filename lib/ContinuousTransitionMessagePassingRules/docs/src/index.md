# ContinuousTransitionMessagePassingRules

The [`ContinuousTransition`](@ref) node and its variational rules. Use it for a linear Gaussian
transition `y ~ N(A x, W⁻¹)` whose matrix `A = f(a)` and noise precision `W` are learned along
with the states: a state-space model with an unknown transition, a rotation by an unknown angle,
a regression with an unknown matrix of coefficients.

```@docs
ContinuousTransitionMessagePassingRules
```

!!! info "Where these rules run"
    [ReactiveMP](https://reactivebayes.github.io/ReactiveMP.jl/dev/) runs these rules on a factor
    graph, which [RxInfer](https://github.com/ReactiveBayes/RxInfer.jl) builds from a
    [GraphPPL](https://github.com/ReactiveBayes/GraphPPL.jl) model. The examples here call the
    rules directly, as a test does.

The site is one page: this one. It follows the node from its definition to its rules, an example
and its limitations, and ends with the API.

## Overview

`ContinuousTransition` carries an `m`-dimensional `x` to an `n`-dimensional `y` through the
`n × m` matrix `f(a)`, with Gaussian noise of precision `W`. The transformation `f` makes the
matrix as free or as structured as the model needs: `a -> reshape(a, n, m)` learns every entry,
while a rotation `a -> [cos(a[1]) -sin(a[1]); sin(a[1]) cos(a[1])]` learns one angle.

The node is a [stochastic node](@extref MessagePassingRulesBase glossary-stochastic-node), a
density over its variables, and its [rules](@extref MessagePassingRulesBase glossary-rule) are
those of [variational message passing](@extref MessagePassingRulesBase glossary-vmp). They update
the [marginals](@extref MessagePassingRulesBase glossary-marginal) `q(a)` and `q(W)` from the
states, and the states from them.

## Definition

```math
p(y \mid x, a, W) = \mathcal{N}\big(y \mid f(a)\, x,\; W^{-1}\big)
```

The output `y` is normal around the transformed input `f(a) x`, with covariance `W⁻¹`.

A nonlinear `f` is linearised around the mean of `q(a)`: each row of `f(a)` is taken as linear
in `a`, through its Jacobian, plus an offset that is zero for an `f` linear through the origin.
The docstring of [`ContinuousTransition`](@ref) gives the expansion and the messages towards `a`
and `W` in full.

## Interfaces

```@example ct
using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily, LinearAlgebra
MessagePassingRulesBase.nodespec(ContinuousTransition)
```

The node has four [interfaces](@extref MessagePassingRulesBase glossary-interface).

| name | meaning | messages and marginals the rules take |
|---|---|---|
| `y` | the output, `n`-dimensional | `MvNormal` message; `MvNormal` joint `q(y, x)` or marginal `q(y)` |
| `x` | the input, `m`-dimensional | `MvNormal` message; `MvNormal` joint `q(y, x)` or marginal `q(x)` |
| `a` | the parameters of `A = f(a)`, a vector | `MvNormal` marginal `q(a)` |
| `W` | the `n × n` noise precision | a marginal with a mean and `E[logdet W]`, such as a `Wishart` |

The rules send an `MvNormal` [message](@extref MessagePassingRulesBase glossary-message)
towards `y`, `x` and `a`, and a Wishart towards `W`.

## Algorithm

```@docs
CTVMP
```

[`CTVMP`](@ref)`(f)` is the node's [algorithm](@extref MessagePassingRulesBase glossary-algorithm).
It takes the transformation as its only argument and has no keywords. The node declares no
algorithm of its own: a model must name `CTVMP` for every `ContinuousTransition`, and under the
default one, [`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm), no rule is
found.

Under `CTVMP` the node declares its
[dependencies](@extref MessagePassingRulesBase glossary-dependencies), the inputs each target's
rule takes:

```@example ct
MessagePassingRulesBase.dependencies_spec(ContinuousTransition, CTVMP(a -> reshape(a, 2, 2)))
```

Every target takes the inputs of the
[default scheme](@extref MessagePassingRulesBase glossary-default-scheme), which follow the
factorisation. The rule towards `a` also reads `q(a)`, the point it expands `f` around.

## Supported rules

```@example ct
MessagePassingRulesBase.rule_coverage(ContinuousTransition)
```

Each target has two rules under `CTVMP`, one per
[factorisation](@extref MessagePassingRulesBase glossary-factorisation):

- the [structured](@extref MessagePassingRulesBase glossary-structured-vmp) `q(y, x) q(a) q(W)`,
  reading the messages on `y` and `x` or the joint `q(y, x)`;
- the [mean field](@extref MessagePassingRulesBase glossary-mean-field) `q(y) q(x) q(a) q(W)`,
  reading marginals only.

The marginal rule computes the joint `q(y, x)`. The
[average energy](@extref MessagePassingRulesBase glossary-average-energy) exists in both
factorisations, so the [Bethe free energy](@extref MessagePassingRulesBase glossary-bethe-free-energy)
is available. There is no [belief propagation](@extref MessagePassingRulesBase glossary-belief-propagation)
rule.

## Example

The message towards `y` under a known identity transition: the mean is `A` times the input's,
the covariance the input's plus the inverse of the mean of `q(W)`. When `a` is uncertain, its
spread adds to the precision of `x` before the transition, as it does in the joint `q(y, x)`.

```@example ct
@call_message_update_rule(
    node = ContinuousTransition, target = :y, algorithm = CTVMP(a -> reshape(a, 2, 2)),
    m = (x = MvNormalMeanCovariance([1.0, 2.0], [1.0 0.0; 0.0 1.0]),),
    q = (a = MvNormalMeanCovariance([1.0, 0.0, 0.0, 1.0], 1e-12 * Matrix(1.0I, 4, 4)), W = Wishart(3, [1.0 0.0; 0.0 1.0])),
)
```

[`@call_message_update_rule`](@extref MessagePassingRulesBase.@call_message_update_rule) runs
the rule and draws its inputs: the message on `x` and the marginals of `a` and `W`. The mean of
`q(W)` is `3 I`, so the covariance is `I + I/3`, which
[`getresult`](@extref MessagePassingRulesBase.getresult) and `mean_cov` confirm:

```jldoctest
julia> using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase, LinearAlgebra

julia> result = @call_message_update_rule(
           node = ContinuousTransition, target = :y, algorithm = CTVMP(a -> reshape(a, 2, 2)),
           m = (x = MvNormalMeanCovariance([1.0, 2.0], [1.0 0.0; 0.0 1.0]),),
           q = (a = MvNormalMeanCovariance([1.0, 0.0, 0.0, 1.0], 1e-12 * Matrix(1.0I, 4, 4)), W = Wishart(3, [1.0 0.0; 0.0 1.0])),
       );

julia> m, V = mean_cov(getresult(result));

julia> m ≈ [1.0, 2.0] && V ≈ (1 + 1 / 3) * I
true
```

## Limitations

- **Variational only.** Every rule needs [`CTVMP`](@ref), which the model must give, and
  marginals on `a` and `W`, which need initialising.
- **Multivariate normals only** on `y`, `x` and `a`: a scalar state is a vector of length one.
- **A nonlinear `f` is linearised**, around the mean of `q(a)`, so its rules and its average
  energy are approximate; for an `f` linear in `a` they are exact.

## API

The node and its alias.

```@docs
ContinuousTransition
CTransition
```
