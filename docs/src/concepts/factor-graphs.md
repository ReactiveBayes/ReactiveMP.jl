# [Factor graphs](@id concepts-factor-graphs)

A **factor graph** is a graphical representation of how a joint probability distribution factorises into a product of local functions. ReactiveMP.jl uses factor graphs as the underlying structure for all inference computations.

## [Variables and factors](@id concepts-factor-graphs-variables-and-factors)

A factor graph has two kinds of nodes:

- **Variable nodes** — represent the random quantities in your model (latent variables, observed data, or constants).
- **Factor nodes** — represent the local functions (conditional distributions, likelihoods, deterministic transforms) that connect variables together.

An edge between a factor node and a variable node means that the factor involves that variable.

Consider a simple model with three variables `x`, `y`, and `z` and two factors `f` and `g`:

```
  (x) ── [f] ── (y) ── [g] ── (z)
```

This graph represents the factorisation:

```math
p(x, y, z) = f(x, y) \cdot g(y, z)
```

Each factor is a *local* function: `f` only involves `x` and `y`, and `g` only involves `y` and `z`. Messages can therefore be computed locally at each factor, using only the information from neighboring nodes.

## [Stochastic and deterministic factors](@id concepts-factor-graphs-node-types)

ReactiveMP.jl distinguishes two kinds of factor nodes:

- [`Stochastic`](@extref MessagePassingRulesBase.Stochastic) nodes are probability
  distributions, such as `p(x | μ, σ)`: likelihoods, priors, and the relations between latent
  variables.
- [`Deterministic`](@extref MessagePassingRulesBase.Deterministic) nodes are functions, such as
  `z = x + y`: they add no probability mass, and enforce an exact relation.

The distinction decides how messages are computed and how the node enters the free energy: a
deterministic node has no average energy, and its clusters are always its output and the joint
over its inputs. See [`sdtype`](@ref), [`isdeterministic`](@ref) and [`isstochastic`](@ref).

## [How ReactiveMP.jl represents factor nodes](@id concepts-factor-graphs-node-registration)

Every factor is a node declared with
[`@define_factor_node`](@extref MessagePassingRulesBase.@define_factor_node) from
`MessagePassingRulesBase`. The declaration names the node, a type or a function, its kind and its
interfaces, the first being the output:

```julia
struct MyFactor end

@define_factor_node(node = MyFactor, type = Stochastic, interfaces = [:out, :x, :y])
```

The declaration gives the node its structure only. Its message update rules, marginal rules and
average energy are defined next to it with the same package's macros, such as
[`@define_message_update_rule`](@extref MessagePassingRulesBase.@define_message_update_rule). The
standard nodes, for common distributions, arithmetic, logic and mixtures, come with
[`StandardMessagePassingRules`](https://reactivebayes.github.io/StandardMessagePassingRules.jl/dev/),
and the others with the packages of [the ecosystem](@ref ecosystem).

In the engine, a node of the graph is a [`FactorNode`](@ref), created with [`factornode`](@ref)
from the node type and the variables it connects.

## [Next steps](@id concepts-factor-graphs-next)

- [Variables](@ref lib-variables): the three kinds of variable and how they work.
- [Message passing](@ref concepts-message-passing): how information flows through the graph.
- [Inference lifecycle](@ref concepts-inference-lifecycle): the phases of building and running
  inference.
