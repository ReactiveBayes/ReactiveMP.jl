# ContinuousTransitionMessagePassingRules

The `ContinuousTransition` node and its variational rules. Use it for a linear Gaussian
transition `y ~ N(A x, W⁻¹)` whose matrix `A = f(a)` and noise precision `W` are learned along
with the states: a state-space model with an unknown transition, a rotation by an unknown angle,
a regression with an unknown matrix of coefficients.

```@docs
ContinuousTransitionMessagePassingRules
```

The site is one page: this one. It follows the node from its definition to its rules, an example
and its limitations, and ends with the API.

## Overview

`ContinuousTransition` carries an `m`-dimensional `x` to an `n`-dimensional `y` through the
`n × m` matrix `f(a)`, with Gaussian noise of precision `W`. The transformation `f` makes the
matrix as free or as structured as the model needs: `a -> reshape(a, n, m)` learns every entry,
while a rotation `a -> [cos(a[1]) -sin(a[1]); sin(a[1]) cos(a[1])]` learns one angle. The node is
variational: its rules update `q(a)` and `q(W)` from the states, and the states from them.

## Definition

```math
p(y \mid x, a, W) = \mathcal{N}\big(y \mid f(a)\, x,\; W^{-1}\big)
```

A nonlinear `f` is linearised around the mean of `q(a)`: each row of `f(a)` is taken as linear
in `a`, through its Jacobian, plus an offset that is zero for an `f` linear through the origin.
The docstring of [`ContinuousTransition`](@ref) gives the expansion and the messages towards `a`
and `W` in full.

## Interfaces

| name | meaning | messages and marginals the rules take |
|---|---|---|
| `y` | the output, `n`-dimensional | `MvNormal` message; `MvNormal` joint `q(y, x)` or marginal `q(y)` |
| `x` | the input, `m`-dimensional | `MvNormal` message; `MvNormal` joint `q(y, x)` or marginal `q(x)` |
| `a` | the parameters of `A = f(a)`, a vector | `MvNormal` marginal `q(a)` |
| `W` | the `n × n` noise precision | a marginal with a mean and `E[logdet W]`, such as a `Wishart` |

The rules send an `MvNormal` towards `y`, `x` and `a`, and a Wishart towards `W`.

## Algorithm

```@docs
CTVMP
```

`CTVMP(f)` takes the transformation as its only argument and has no keywords. The node declares
no algorithm of its own: a model must name `CTVMP` for every `ContinuousTransition`, and under the
default one, [`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm), no rule is
found. Its rule towards `a` reads `q(a)`, the point it expands `f` around, besides the inputs its
factorisation gives it.

## Supported rules

```@example
using MessagePassingRulesBase, ContinuousTransitionMessagePassingRules # hide
MessagePassingRulesBase.rule_coverage(ContinuousTransition)
```

Each target has two rules under `CTVMP`: one for the structured factorisation
`q(y, x) q(a) q(W)`, reading the messages on `y` and `x` or the joint `q(y, x)`, and one for the
mean field `q(y) q(x) q(a) q(W)`, reading marginals only. The marginal rule computes the joint
`q(y, x)`, and the average energy exists in both factorisations, so the Bethe free energy is
available. There is no belief-propagation rule.

## Example

The message towards `y` under a known identity transition: the mean is `A` times the input's,
the covariance the input's plus the inverse of the mean of `q(W)`.

```jldoctest
julia> using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase, LinearAlgebra

julia> result = @call_message_update_rule(
           node = ContinuousTransition, target = :y, algorithm = CTVMP(a -> reshape(a, 2, 2)),
           m = (x = MvNormalMeanCovariance([1.0, 2.0], [1.0 0.0; 0.0 1.0]),),
           q = (a = MvNormalMeanCovariance([1.0, 0.0, 0.0, 1.0], 1e-8 * Matrix(1.0I, 4, 4)), W = Wishart(3, [1.0 0.0; 0.0 1.0])),
       );

julia> m, V = mean_cov(getresult(result));

julia> m ≈ [1.0, 2.0] && V ≈ (1 + 1 / 3) * I
true
```

In a model, with RxInfer, the algorithm names the transformation, and the factorisation is
structured over each transition's `y` and `x`:

```julia
using RxInfer, ContinuousTransitionMessagePassingRules

@model function rotating_state(y, x0)
    a ~ MvNormal(mean = zeros(1), covariance = [1.0;;])
    W ~ Wishart(3, diageye(2))
    x_prev = x0
    for i in eachindex(y)
        x[i] ~ ContinuousTransition(x_prev, a, W)
        y[i] ~ MvNormal(mean = x[i], covariance = diageye(2))
        x_prev = x[i]
    end
end

@algorithm function rotation_algorithm()
    ContinuousTransition() -> CTVMP(a -> [cos(a[1]) -sin(a[1]); sin(a[1]) cos(a[1])])
end

@constraints function rotation_constraints()
    q(x0, x, a, W) = q(x0, x)q(a)q(W)
end

@initialization function rotation_init()
    q(a) = MvNormalMeanCovariance(zeros(1), [1.0;;])
    q(W) = Wishart(3, diageye(2))
end
```

## Limitations

- **Variational only.** Every rule needs `CTVMP`, which the model must give, and marginals on
  `a` and `W`, which need initialising.
- **Multivariate normals only** on `y`, `x` and `a`: a scalar state is a vector of length one.
- **A nonlinear `f` is linearised**, around the mean of `q(a)`, so its rules and its average
  energy are approximate; for an `f` linear in `a` they are exact.

## API

The node and its alias.

```@docs
ContinuousTransition
CTransition
```
