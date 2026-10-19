# [Factor graphs](@id concepts-factor-graphs)

A [factor graph](@extref MessagePassingRulesBase glossary-factor-graph) draws how a joint
probability density factorises into a product of local functions, the factors. ReactiveMP.jl
runs all its inference on a factor graph.

```@setup factor-graphs
using ReactiveMP, MessagePassingRulesBase, BayesBase, ExponentialFamily
include(joinpath(pkgdir(ReactiveMP), "docs", "nodes.jl"))
```

## [Variables and factors](@id concepts-factor-graphs-variables-and-factors)

A factor graph has two kinds of node:

- A **variable** stands for one quantity of the model: a latent quantity you infer, an
  observation, or a constant.
- A **factor node** stands for one local function: a prior, a likelihood, a conditional
  distribution or a deterministic relation. Its edges are its
  [interfaces](@extref MessagePassingRulesBase glossary-interface).

An edge joins a factor node to a variable when the factor's function takes that variable as an
argument. Take the model of [Getting started](@ref getting-started), a latent `x` observed through
normal noise:

```math
p(x, y) = \underbrace{\mathcal{N}(x \mid 0, 10)}_{f(x)} \; \underbrace{\mathcal{N}(y \mid x, 1)}_{g(x, y)}.
```

It has two factors and two variables:

```
  [f] ──── (x) ──── [g] ──── (y)
```

Each factor is *local*: `f` takes only `x`, and `g` takes only `x` and `y`. A message along an
edge is therefore computed at one node, from what arrives at that node alone (see
[Message passing](@ref concepts-message-passing)).

## [The graph in code](@id concepts-factor-graphs-code)

In ReactiveMP.jl every argument of a factor is a variable, the known ones included: the mean
``0`` and the variances ``10`` and ``1`` are constants on edges of their own. Both factors are the
`Gaussian` node of [the example node](@ref example-node), a normal density with interfaces `out`,
`μ` and `v`:

```@example factor-graphs
x = randomvar(label = :x)
y = datavar(label = :y)

f = factornode(Gaussian, [(:out, x), (:μ, constvar(0.0)), (:v, constvar(10.0))])
g = factornode(Gaussian, [(:out, y), (:μ, x), (:v, constvar(1.0))])
```

A [`randomvar`](@ref) is inferred, a [`datavar`](@ref) receives observations and a
[`constvar`](@ref) holds a fixed value (see [Variables](@ref lib-variables)). A
[`factornode`](@ref) creates a [`FactorNode`](@ref), connecting each interface to one variable.
The variable `x` has two edges, one to each factor:

```@example factor-graphs
ReactiveMP.degree(x)
```

## [Stochastic and deterministic factors](@id concepts-factor-graphs-node-types)

A factor node is one of two kinds:

- A [stochastic node](@extref MessagePassingRulesBase glossary-stochastic-node),
  [`Stochastic`](@extref MessagePassingRulesBase.Stochastic), is a probability density over its
  interfaces, such as ``\mathcal{N}(y \mid x, 1)``: a prior, a likelihood, or the relation
  between latent variables.
- A [deterministic node](@extref MessagePassingRulesBase glossary-deterministic-node),
  [`Deterministic`](@extref MessagePassingRulesBase.Deterministic), is a function, such as
  ``z = x + c``: it enforces an exact relation and adds no uncertainty.

`Gaussian` is stochastic. A deterministic node is declared the same way, with
`type = Deterministic`; this one computes `out = in + c`:

```@example factor-graphs
struct Shift end

@define_factor_node(node = Shift, type = Deterministic, interfaces = [:out, :in, :c])

z = randomvar(label = :z)
factornode(Shift, [(:out, z), (:in, x), (:c, constvar(1.0))])
```

The kind decides how messages are computed and how the node enters the free energy. A
deterministic node has no average energy. Its clusters are always its output and the joint over
its inputs, `(out)` and `(in, c)` here. [`sdtype`](@ref), [`isdeterministic`](@ref) and
[`isstochastic`](@ref) read the kind back.

## [Where nodes come from](@id concepts-factor-graphs-node-registration)

The engine defines no node. A node is declared with
[`@define_factor_node`](@extref MessagePassingRulesBase.@define_factor_node) from
[`MessagePassingRulesBase`](@extref MessagePassingRulesBase MessagePassingRulesBase), as `Shift`
is above. The declaration names the node, its kind and its interfaces, the first being the
output. It is data, which draws itself:

```@example factor-graphs
MessagePassingRulesBase.nodespec(Gaussian)
```

The declaration gives the node its structure only. Its message update rules, marginal rules and
average energy are defined next to it with the same package's macros, such as
[`@define_message_update_rule`](@extref MessagePassingRulesBase.@define_message_update_rule);
[the example node](@ref example-node) shows them for `Gaussian`. The standard nodes, for the
common distributions, arithmetic, logic and mixtures, are in
[`StandardMessagePassingRules`](@extref StandardMessagePassingRules StandardMessagePassingRules),
and the others in the packages of [the ecosystem](@ref ecosystem).

## [Next steps](@id concepts-factor-graphs-next)

- [Variables](@ref lib-variables): the three kinds of variable and how they work.
- [Message passing](@ref concepts-message-passing): how information flows through the graph.
- [Inference lifecycle](@ref concepts-inference-lifecycle): the phases of building and running
  inference.
