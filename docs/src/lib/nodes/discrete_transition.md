# [Discrete transition node](@id lib-nodes-discrete-transition)

The `DiscreteTransition` node encodes a **Markov state transition** for discrete categorical variables. It represents the conditional distribution:

```math
p(\text{out} \mid \text{in}, A) = A \cdot \text{in}
```

where `out` and `in` are categorical (discrete) state variables and `A` is a column-stochastic transition matrix (each column sums to one). This is the fundamental building block for **Hidden Markov Models (HMMs)** and other discrete state-space models.

## [Interfaces](@id lib-nodes-discrete-transition-interfaces)

The `DiscreteTransition` node accepts a variable number of inputs:

| Interface index | Alias | Role |
|----------------|-------|------|
| 1 | `out` | Next (output) state |
| 2 | `in` | Current (input) state |
| 3 | `a` | Transition matrix variable |
| 4, 5, … | `T1`, `T2`, … | Optional additional transition matrices |

This flexible interface allows multi-dimensional transition structures where the full transition is a product of several matrices.

## [Comparison with a plain Categorical node](@id lib-nodes-discrete-transition-vs-categorical)

A plain `Categorical` node fixes the probability vector at the time the node is created. `DiscreteTransition` is different in two important ways:

1. **The transition matrix `a` is a variable** — it can have a `DirichletCollection` prior and its posterior is inferred jointly with the states.
2. **The input state is also a variable** — messages flow in both directions, making it possible to infer both past states (smoothing) and future states (prediction).

## [Typical usage pattern](@id lib-nodes-discrete-transition-usage)

```julia
# prior on initial state
s[1] ~ Categorical(fill(1/K, K))

# prior on transition matrix (one Dirichlet per column)
A ~ DirichletCollection(ones(K, K))

# Markov chain
for t in 2:T
    s[t] ~ DiscreteTransition(s[t-1], A)
end

# emission likelihoods
for t in 1:T
    y[t] ~ Categorical(B * s[t])
end
```

## [Utility functions](@id lib-nodes-discrete-transition-utils)

The following internal functions implement the message update rules for the `DiscreteTransition` node. They are exposed for users who want to reuse them in custom rule definitions.

```@docs
ReactiveMP.discrete_transition_decode_marginal
ReactiveMP.discrete_transition_marginal_rule
ReactiveMP.discrete_transition_process_marginals
ReactiveMP.multiply_dimensions!
ReactiveMP.sum_out_dimensions
ReactiveMP.discrete_transition_process_messages
ReactiveMP.discrete_transition_structured_message_rule
```
