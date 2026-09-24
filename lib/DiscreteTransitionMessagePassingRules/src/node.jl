"""
    DiscreteTransition

The node `out ~ Categorical(A[:, in, T1, …, Tn])`: a transition from `in`, conditioned on any
number of categoricals `T`, through a tensor `A` of probabilities with axes `(out, in, T1, …, Tn)`,
whose marginal `q(a)` is a `DirichletCollection` or a point mass. Its interfaces are `out`, `in`,
`a` and the group `T`, which may be empty.

A tensor node: each of its rules is one contraction of `E[log A]` with its inputs along the axes
they cover, written once for any factorisation. Belief propagation, mean-field, structured
variational message passing and joints of some of the `T`s are the same rules with other inputs.
`q(a)` is kept in a cluster of its own. It runs under `DefaultAlgorithm`.
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
