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

Every factor is a node declared with `@define_factor_node` from `MessagePassingRulesBase`. The
declaration names the node, a type or a function, its type (`Stochastic` or `Deterministic`)
and its interfaces, the first being the output:

```julia
struct MyFactor end

@define_factor_node(node = MyFactor, type = Stochastic, interfaces = [:out, :x, :y])
```

The declaration gives the node its structure only. Its message update rules, marginal rules and
average energy are defined next to it with the same package's macros; see
[Defining nodes and rules](@ref rules-defining). The standard nodes, for common distributions,
arithmetic, logic and mixtures, come with `StandardMessagePassingRules`
([Standard rules](@ref packages-standard)).

## [Next steps](@id concepts-factor-graphs-next)

- [Variables](@ref lib-variables) — the three kinds of variable nodes and how they work.
- [Message passing](@ref concepts-message-passing) — how information flows through the graph.
- [Inference lifecycle](@ref concepts-inference-lifecycle) — the three phases of building and running inference.
