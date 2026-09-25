# [DiscreteTransition](@id packages-discrete-transition)

```@docs
DiscreteTransitionMessagePassingRules
```

`DiscreteTransition` is `out ~ Categorical(A[:, in, T1, …, Tn])`: a transition from the categorical
`in` to the categorical `out`, conditioned on any number of categoricals `T`, through a tensor `A`
of probabilities with axes `(out, in, T1, …, Tn)`. Its interfaces are `out`, `in`, `a` and the
group `T`, which may be empty, so the `k`-th conditioning variable is the member `(:T, k)`. `q(a)`
is a `DirichletCollection`, when `A` is learned, or a point mass of the tensor.

```@docs
DiscreteTransition
```

## [A tensor node](@id packages-discrete-transition-tensor)

Every rule of the node is one contraction of `E[log A]` with its inputs, each along the tensor
axes it covers: `out` is axis 1, `in` axis 2, `(:T, k)` axis `2 + k`, and a joint the axes of its
members, in order. The inputs that are marginals are summed out of `E[log A]`, which is then
exponentiated, and the messages multiply the result along their axes. The rules take whatever
inputs the factorisation delivers (`default` in their arguments, see
[Rules over whatever the factorisation delivers](@ref rules-defining-default-args)), so each is written
once, for any number of `T`s and any factorisation:

- belief propagation, every categorical interface one cluster with the others;
- mean-field, `q(out) q(in) q(T1) … q(a)`;
- structured factorisations, such as `q(out, in) q(a)` in a hidden Markov model;
- joints of some of the `T`s, such as `q(out, T1) q(in) q(T2) q(a)`, whose key is
  `(:out, (:T, 1))`.

The message towards `a` is the expected counts, the outer product of the marginals of the other
interfaces along their axes, plus one. The marginal of a cluster splits off each member observed
as a point mass that is a member of the cluster in its own right, as a
[`FactorizedCluster`](@ref) block; one inside a whole group `T` stays in the joint, as a one-hot
axis.

The contractions are two helpers, public, and called qualified:

```@docs
DiscreteTransitionMessagePassingRules.multiply_dimensions!
DiscreteTransitionMessagePassingRules.sum_out_dimensions
```

A joint's axes are read from its key, never from a name, and every message is normalised over
the whole tensor. There are no rules specialised to a number of interfaces: the generic
contraction is within 1.0–1.6 times the cost of a specialised two-interface rule from ten states
on, and 90 ns slower at two.

The model layer names the interfaces, and the conditioning variables are the members
`(:T, 1)`, `(:T, 2)`.
