@doc raw"""
    DiscreteTransition

The stochastic node of a transition between categoricals through a tensor of probabilities,

```math
p(out \mid in, T_1, \ldots, T_n, A) = A[out, in, T_1, \ldots, T_n],
\qquad \textstyle\sum_{i} A[i, j, t_1, \ldots, t_n] = 1,
```

that is `out ~ Categorical(A[:, in, T1, …, Tn])`: a transition from `in` to `out`, conditioned on
any number of categoricals `T`, with the tensor `A` known or learned.

# Interfaces

- `out`: the next category, axis 1 of `A`;
- `in`: the current category, axis 2;
- `a`: the tensor `A`, whose marginal `q(a)` is a `DirichletCollection` when `A` is learned, or a
  `PointMass` of the tensor when it is known;
- `T`: a group of conditioning categoricals, possibly empty; its `k`-th member, `(:T, k)`, is
  axis `2 + k`.

The categorical interfaces take `Categorical`, `Bernoulli` or `PointMass` (one-hot) messages and
marginals, and a joint over some of them is a `Contingency` with its members' axes in order.

# A tensor node

Each rule is one contraction of `E[log A]` with its inputs, each along the axes it covers. The
inputs that are marginals are summed out of `E[log A]`, which is then exponentiated, and the
messages multiply the result along their axes: the message towards `out` under belief
propagation, with `q(a)` a point mass, is `Σⱼ A[:, j] m_in(j)`, and under mean field
`exp(Σⱼ E[log A[:, j]] q_in(j))`. Every rule takes whatever inputs the factorisation delivers, so
each is written once for any number of `T`s and any factorisation: belief propagation,
mean-field, structured ones such as `q(out, in) q(a)` in a hidden Markov model, and joints of some
of the `T`s, such as `q(out, (T, 1)) q(in) q(a)`. `q(a)` is always a cluster of its own.

The message towards `a` is the expected counts plus one: a `DirichletCollection` with

```math
\alpha[i, j, t_1, \ldots] = 1 + \mathbb{E}_q\big[\mathbb{1}[out = i, in = j, T_1 = t_1, \ldots]\big],
```

the outer product of the marginals of the other interfaces along their axes, or the joint's
tensor for a joint, plus one, so that its product with a `DirichletCollection(α₀)` prior adds
the expected counts to `α₀`.

The marginal of a cluster multiplies the exponentiated tensor by its members' messages. A member
observed as a `PointMass` that is a member of the cluster in its own right is split off, as a
[`FactorizedCluster`](@extref MessagePassingRulesBase.FactorizedCluster) block; one inside a
whole group `T` stays in the joint, as a one-hot axis.

The average energy is `-Σ E[log A] ⊙ q`, `E[log A]` weighted by every marginal along its axes.

# Algorithm

The node runs under the default algorithm,
[`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm), which a model does not
name.

# Limitations

- `q(a)` must be a `DirichletCollection` or a `PointMass` of an array: the rules towards the
  categoricals, the marginals and the average energy take no other.
- Every message is normalised over the whole tensor, and `E[log A]` is clamped away from
  `log 0`, so a zero probability becomes a tiny one.

# Examples

A known two-state transition, the message towards `out` from a uniform message on `in`:

```jldoctest; setup = :(using DiscreteTransitionMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily)
julia> A = [0.9 0.2; 0.1 0.8];

julia> result = @call_message_update_rule(
           node = DiscreteTransition, target = :out,
           m = (in = Categorical([0.5, 0.5]),), q = (a = PointMass(A),),
       );

julia> probvec(getresult(result)) ≈ [0.55, 0.45]
true
```

The message towards `a` from mean-field marginals, the expected counts plus one:

```jldoctest; setup = :(using DiscreteTransitionMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily)
julia> result = @call_message_update_rule(
           node = DiscreteTransition, target = :a,
           q = (out = Categorical([0.2, 0.8]), in = Categorical([1.0, 0.0])),
       );

julia> params(getresult(result))[1] ≈ [1.2 1.0; 1.8 1.0]
true
```
"""
struct DiscreteTransition end

@define_factor_node(node = DiscreteTransition, type = Stochastic, interfaces = [:out, :in, :a, :T...], min_group_length = 0)

# The tensor axes an input covers: `out` is 1, `in` 2, the `k`-th of `T` 2 + k, and a whole group
# `T` inside a joint every axis of the tensor after the others; a joint, its members' in order.
# `n` is the number of axes the input's own tensor has, which a whole group needs.
discrete_transition_axes(key::Symbol, n) = key === :out ? (1,) : key === :in ? (2,) : key === :T ? ntuple(i -> 2 + i, n) :
    throw(ArgumentError("`$key` is not a categorical interface of `DiscreteTransition`"))
discrete_transition_axes(key::Tuple{Symbol, Int}, n) = first(key) === :T ? (2 + last(key),) :
    throw(ArgumentError("`$(first(key))` is not a group of `DiscreteTransition`"))
function discrete_transition_axes(key::Tuple, n)
    others = count(member -> member !== :T, key)
    return Tuple(Iterators.flatten(map(member -> discrete_transition_axes(member, member === :T ? n - others : 1), key)))
end

# An input as the tensor it weights with: a probability vector, or a joint's tensor.
discrete_transition_weights(d::DiscreteNonParametric) = probvec(d)
discrete_transition_weights(d::Bernoulli) = collect(probvec(d))
discrete_transition_weights(d::PointMass{<:AbstractArray}) = mean(d)
discrete_transition_weights(d::Contingency) = components(d)

const DiscreteTransitionTensor = Union{DirichletCollection, PointMass{<:AbstractArray}}
