# [Factor graphs](@id concepts-factor-graphs)

A **factor graph** is a graphical representation of how a joint probability distribution factorizes into a product of local functions. ReactiveMP.jl uses factor graphs as the underlying structure for all inference computations.

## [Variables and factors](@id concepts-factor-graphs-variables-and-factors)

A factor graph has two kinds of nodes:

- **Variable nodes** — represent the random quantities in your model (latent variables, observed data, or constants).
- **Factor nodes** — represent the local functions (conditional distributions, likelihoods, deterministic transforms) that connect variables together.

An edge between a factor node and a variable node means that the factor involves that variable.

Consider a simple model with three variables `x`, `y`, and `z` and two factors `f` and `g`:

```
  (x) ── [f] ── (y) ── [g] ── (z)
```

This graph represents the factorization:

```math
p(x, y, z) = f(x, y) \cdot g(y, z)
```

Each factor is a *local* function: `f` only involves `x` and `y`, and `g` only involves `y` and `z`. Messages can therefore be computed locally at each factor, using only the information from neighboring nodes.

## [Stochastic and deterministic factors](@id concepts-factor-graphs-node-types)

ReactiveMP.jl distinguishes two kinds of factor nodes:

- [`Stochastic`](@ref) nodes represent probability distributions, e.g. `p(x | μ, σ)`. They are used for likelihood terms, priors, and latent variable relationships.
- [`Deterministic`](@ref) nodes represent hard functional constraints, e.g. `z = x + y`. They do not add probability mass — they enforce an exact relationship.

This distinction matters for how messages are computed and how the variational free energy objective is structured. See [`isdeterministic`](@ref) and [`isstochastic`](@ref).

## [How ReactiveMP.jl represents factor nodes](@id concepts-factor-graphs-node-registration)

Every factor in ReactiveMP.jl is a Julia type registered with the [`@node`](@ref) macro. The macro declares the node's name, its type (`Stochastic` or `Deterministic`), and the fixed set of edges (interfaces) it connects to:

```julia
struct MyFactor end

@node MyFactor Stochastic [ out, x, y ]
#     ^^^^^^^^ ^^^^^^^^^^   ^^^^^^^^^^
#     tag      type          edges (first = output by convention)
```

After registration, `MyFactor` can be used as a factor node in a model. The inference engine then dispatches message update rules defined with [`@rule`](@ref) for that node type.

!!! note
    The `@node` macro only registers the factor's structure. Message update rules must be added separately using [`@rule`](@ref) and [`@marginalrule`](@ref). See [Message update rules](@ref lib-rules) for details.

ReactiveMP.jl ships with [many predefined nodes](@ref lib-predefined-nodes) for common distributions and operations — Gaussian, Gamma, Beta, Bernoulli, arithmetic operations, and more. Custom nodes can be registered using the same `@node` macro.

## [Next steps](@id concepts-factor-graphs-next)

- [Variables](@ref lib-variables) — the three kinds of variable nodes and how they work.
- [Message passing](@ref concepts-message-passing) — how information flows through the graph.
- [Inference lifecycle](@ref concepts-inference-lifecycle) — the three phases of building and running inference.
