# DiscreteTransitionMessagePassingRules

The `DiscreteTransition` node: a transition between categoricals through a tensor of
probabilities, conditioned on any number of other categoricals. Use it for the transition and
emission factors of hidden Markov models, for a known or a learned transition matrix, and for
transitions that switch with a regime or an action.

```@docs
DiscreteTransitionMessagePassingRules
```

The site has two pages: this one, the node from its definition to its API, and
[Internals](internals.md), the tensor algebra its rules are built on.

## Overview

`DiscreteTransition` takes the category of `in` to the category of `out` through the column
`A[:, in]` of a transition matrix, or, with conditioning categoricals `T1, …, Tn`, through the
fibre `A[:, in, T1, …, Tn]` of a tensor. `A` is known, as a `PointMass`, or learned, with a
`DirichletCollection` prior. The node is a *tensor node*: each of its rules is one contraction of
the tensor with its inputs, so the same rules serve belief propagation, mean-field and structured
variational message passing, for any number of `T`s.

## Definition

```math
p(out \mid in, T_1, \ldots, T_n, A) = A[out, in, T_1, \ldots, T_n],
\qquad \textstyle\sum_{i} A[i, j, t_1, \ldots, t_n] = 1
```

The axes of `A` are `(out, in, T1, …, Tn)`: `out` is axis 1, `in` axis 2 and `(:T, k)` axis
`2 + k`. A joint over some of the interfaces covers its members' axes, in order.

## Interfaces

| name | meaning | messages and marginals the rules take |
|---|---|---|
| `out` | the next category, axis 1 | `Categorical`, `Bernoulli` or one-hot `PointMass`; in a joint, `Contingency` |
| `in` | the current category, axis 2 | as `out` |
| `a` | the tensor `A` | `DirichletCollection` or `PointMass` of an array, always its own cluster |
| `T` | a group of conditioning categoricals, possibly empty; `(:T, k)` is axis `2 + k` | as `out` |

The rules send a `Categorical` towards `out`, `in` and each `(:T, k)`, a `DirichletCollection`
towards `a`, and a `Categorical`, a `Contingency` or a
[`FactorizedCluster`](@extref MessagePassingRulesBase.FactorizedCluster) as the marginal of a
cluster.

## Algorithm

The node runs under the default algorithm,
[`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm), which has no parameters,
and a model names none.

## Supported rules

| target | rules under `DefaultAlgorithm` |
|---|---|
| → out | ✓ |
| → in | ✓ |
| → a | ✓ |
| → (T, k) | ✓ |
| the marginal of any cluster | ✓ |
| average energy | ✓ |

Each row is one rule, over whatever inputs the factorisation delivers: belief propagation, every
categorical interface in one cluster with the others; mean field,
`q(out) q(in) q(T1) … q(a)`; structured factorisations such as `q(out, in) q(a)`; and joints of
some of the `T`s, such as `q(out, (T, 1)) q(in) q(a)`. With the average energy, the Bethe free
energy is available in every factorisation. (The table is written by hand: the node's marginal
rule covers every cluster, which `rule_coverage` does not display.)

### How the rules contract

Every rule is a contraction of `E[log A]` with its inputs, each along the axes it covers. The
marginals are summed out of `E[log A]`, which is then exponentiated, and the messages multiply the
result along their axes. With `q(a)` a point mass and nothing to sum out, the tensor is `A`
itself. So the message towards `out` is `Σⱼ A[:, j] m_in(j)` under belief propagation, and
`exp(Σⱼ E[log A[:, j]] q_in(j))` under mean field.

The message towards `a` is the expected counts plus one:

```math
\alpha[i, j, t_1, \ldots] = 1 + \mathbb{E}_q\big[\mathbb{1}[out = i, in = j, T_1 = t_1, \ldots]\big],
```

the outer product of the marginals of the other interfaces, or a joint's tensor, plus one; its
product with a `DirichletCollection(α₀)` prior adds the expected counts to `α₀`.

The marginal of a cluster multiplies the exponentiated tensor by its members' messages. A member
observed as a `PointMass` that is a member of the cluster in its own right is split off, as a
[`FactorizedCluster`](@extref MessagePassingRulesBase.FactorizedCluster) block; one inside a whole
group `T` stays in the joint, as a one-hot axis.

## Example

A known two-state transition. The message towards `out` from a uniform message on `in` is the
average of the columns of `A`; the message towards `a` from mean-field marginals is the expected
counts plus one.

```jldoctest
julia> using DiscreteTransitionMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily

julia> A = [0.9 0.2; 0.1 0.8];

julia> result = @call_message_update_rule(
           node = DiscreteTransition, target = :out,
           m = (in = Categorical([0.5, 0.5]),), q = (a = PointMass(A),),
       );

julia> probvec(getresult(result)) ≈ [0.55, 0.45]
true

julia> result = @call_message_update_rule(
           node = DiscreteTransition, target = :a,
           q = (out = Categorical([0.2, 0.8]), in = Categorical([1.0, 0.0])),
       );

julia> params(getresult(result))[1] ≈ [1.2 1.0; 1.8 1.0]
true
```

In a model, with RxInfer, a hidden Markov model with learned transition and emission matrices,
under the structured factorisation over the states:

```julia
using RxInfer, DiscreteTransitionMessagePassingRules

@model function hidden_markov_model(x)
    A ~ DirichletCollection(ones(3, 3))
    B ~ DirichletCollection([10.0 1.0 1.0; 1.0 10.0 1.0; 1.0 1.0 10.0])
    s_0 ~ Categorical(fill(1.0 / 3.0, 3))
    s_prev = s_0
    for t in eachindex(x)
        s[t] ~ DiscreteTransition(s_prev, A)
        x[t] ~ DiscreteTransition(s[t], B)
        s_prev = s[t]
    end
end

@constraints function hidden_markov_model_constraints()
    q(s_0, s, A, B) = q(s_0, s)q(A)q(B)
end
```

A conditioning categorical is passed after the tensor, `s[t] ~ DiscreteTransition(s_prev, A,
u[t])`, and is the member `(:T, 1)`.

## Limitations

- `q(a)` must be a `DirichletCollection` or a `PointMass` of an array.
- `E[log A]` is clamped away from `log 0`, so an impossible transition is given a tiny
  probability rather than zero.
- The contractions are generic, for any number of interfaces: none is specialised to the
  two-interface case.

## API

```@docs
DiscreteTransition
```
