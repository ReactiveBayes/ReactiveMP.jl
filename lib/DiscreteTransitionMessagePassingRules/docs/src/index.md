# DiscreteTransitionMessagePassingRules

The [`DiscreteTransition`](@ref) node: a transition between categorical variables through a
tensor of probabilities, conditioned on any number of other categorical variables. Use it for
the transition and emission factors of hidden Markov models, for a known or a learned transition
matrix, and for transitions that switch with a regime or an action.

```@docs
DiscreteTransitionMessagePassingRules
```

!!! info "Where these rules run"
    [ReactiveMP](https://reactivebayes.github.io/ReactiveMP.jl/dev/) runs these rules on a factor
    graph, which [RxInfer](https://github.com/ReactiveBayes/RxInfer.jl) builds from a
    [GraphPPL](https://github.com/ReactiveBayes/GraphPPL.jl) model. The examples here call the
    rules directly, as a test does.

The site has two pages: this one, the node from its definition to its API, and
[Internals](internals.md), the tensor algebra its rules are built on.

## Overview

A hidden Markov model moves between a finite number of states. The probability of the next
state given the current one is a column of a transition matrix `A`. `DiscreteTransition` is the
[factor node](@extref MessagePassingRulesBase glossary-factor-node) of that step: it takes the
category of `in` to the category of `out` through the column `A[:, in]`. With conditioning
categoricals `T1, …, Tn`, it takes the fibre `A[:, in, T1, …, Tn]` of a tensor instead, so the
transition can switch with a regime or an action.

`A` is known, as a [point mass](@extref MessagePassingRulesBase glossary-point-mass), or learned,
with a `DirichletCollection` prior. The node is a *tensor node*: each of its
[rules](@extref MessagePassingRulesBase glossary-rule) is one contraction of the tensor with its
inputs. The same rules therefore serve
[belief propagation](@extref MessagePassingRulesBase glossary-belief-propagation),
[mean-field](@extref MessagePassingRulesBase glossary-mean-field) and
[structured](@extref MessagePassingRulesBase glossary-structured-vmp) variational message
passing, for any number of `T`s.

## Definition

```math
p(out \mid in, T_1, \ldots, T_n, A) = A[out, in, T_1, \ldots, T_n],
\qquad \textstyle\sum_{i} A[i, j, t_1, \ldots, t_n] = 1
```

The probability that `out` is category `i` is the entry of `A` at `i` and the categories of the
other variables. The constraint makes every fibre `A[:, j, t₁, …, tₙ]` a probability vector over
`out`. The axes of `A` are `(out, in, T1, …, Tn)`: `out` is axis 1, `in` axis 2 and `(:T, k)`
axis `2 + k`. A joint over some of the interfaces covers its members' axes, in order.

## Interfaces

```@example dt
using DiscreteTransitionMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily
MessagePassingRulesBase.nodespec(DiscreteTransition)
```

The node has three [interfaces](@extref MessagePassingRulesBase glossary-interface) and a
[group](@extref MessagePassingRulesBase glossary-group) `T` of any length, zero included.

| name | meaning | messages and marginals the rules take |
|---|---|---|
| `out` | the next category, axis 1 | `Categorical`, `Bernoulli` or one-hot `PointMass`; in a joint, `Contingency` |
| `in` | the current category, axis 2 | as `out` |
| `a` | the tensor `A` | `DirichletCollection` or `PointMass` of an array, always its own cluster |
| `T` | a group of conditioning categoricals, possibly empty; `(:T, k)` is axis `2 + k` | as `out` |

The rules send a `Categorical` towards `out`, `in` and each `(:T, k)`, and a
`DirichletCollection` towards `a`. As the
[marginal](@extref MessagePassingRulesBase glossary-marginal) of a
[cluster](@extref MessagePassingRulesBase glossary-cluster) they return a `Categorical`, a
`Contingency` or a [`FactorizedCluster`](@extref MessagePassingRulesBase.FactorizedCluster).

## Algorithm

The node runs under the default [algorithm](@extref MessagePassingRulesBase glossary-algorithm),
[`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm), which has no parameters,
and a model names none. The node declares no
[dependencies](@extref MessagePassingRulesBase glossary-dependencies): each rule takes the inputs
of the [default scheme](@extref MessagePassingRulesBase glossary-default-scheme).

## Supported rules

```@example dt
MessagePassingRulesBase.rule_coverage(DiscreteTransition)
```

[`rule_coverage`](@extref MessagePassingRulesBase.rule_coverage) shows one rule per target. Each
rule takes whatever inputs the [factorisation](@extref MessagePassingRulesBase glossary-factorisation)
delivers:

- belief propagation, with every categorical interface in one cluster;
- mean field, `q(out) q(in) q(T1) … q(a)`;
- structured factorisations such as `q(out, in) q(a)`;
- joints of some of the `T`s, such as `q(out, (T, 1)) q(in) q(a)`.

The row `q(any cluster)` is one marginal rule for every cluster. With the
[average energy](@extref MessagePassingRulesBase glossary-average-energy), the
[Bethe free energy](@extref MessagePassingRulesBase glossary-bethe-free-energy) is available in
every factorisation.

### How the rules contract

Every rule is a contraction of `E[log A]` with its inputs, each along the axes it covers. The
marginals are summed out of `E[log A]`, which is then exponentiated, and the
[messages](@extref MessagePassingRulesBase glossary-message) multiply the result along their
axes. With `q(a)` a point mass and nothing to sum out, the tensor is `A` itself. So the message
towards `out` is `Σⱼ A[:, j] m_in(j)` under belief propagation, and
`exp(Σⱼ E[log A[:, j]] q_in(j))` under mean field.

The message towards `a` is the expected counts plus one:

```math
\alpha[i, j, t_1, \ldots] = 1 + \mathbb{E}_q\big[\mathbb{1}[out = i, in = j, T_1 = t_1, \ldots]\big].
```

The expectation is the outer product of the marginals of the other interfaces, or a joint's
tensor. The product of this message with a `DirichletCollection(α₀)` prior adds the expected
counts to `α₀`.

The marginal of a cluster multiplies the exponentiated tensor by its members' messages. A member
observed as a `PointMass` that is a member of the cluster in its own right is split off, as a
[`FactorizedCluster`](@extref MessagePassingRulesBase.FactorizedCluster) block. A member inside a
whole group `T` stays in the joint, as a one-hot axis.

## Example

A known two-state transition, and the message towards `out` from a uniform message on `in`:

```@example dt
A = [0.9 0.2; 0.1 0.8]

@call_message_update_rule(
    node = DiscreteTransition, target = :out,
    m = (in = Categorical([0.5, 0.5]),), q = (a = PointMass(A),),
)
```

[`@call_message_update_rule`](@extref MessagePassingRulesBase.@call_message_update_rule) runs
the rule and draws the inputs it read. The result is the average of the columns of `A`.
[`getresult`](@extref MessagePassingRulesBase.getresult) extracts the distribution, and the
message towards `a` from mean-field marginals is the expected counts plus one:

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
