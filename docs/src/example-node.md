# [The example node](@id example-node)

ReactiveMP defines no factor node. Nodes and their rules come from rule packages, such as
[StandardMessagePassingRules](@extref StandardMessagePassingRules StandardMessagePassingRules),
which hold the distributions and functions a model uses. So that its examples load none of them,
this site declares one small node with
[MessagePassingRulesBase](https://reactivebayes.github.io/MessagePassingRulesBase.jl/dev/), as a
rule package would: `Gaussian`, a normal density with a known variance,

```math
f(\text{out}, \mu, v) = \mathcal{N}(\text{out} \mid \mu, v).
```

It has the belief propagation rules towards `out` and `μ`, the joint marginal of its cluster, the
variational rules for independent marginals, and its average energy. Every page includes the
file below with

```julia
include(joinpath(pkgdir(ReactiveMP), "docs", "nodes.jl"))
```

The tutorial [Your first node](@extref MessagePassingRulesBase tutorial-first-node) builds a node
like it step by step.

```@eval
using Markdown, ReactiveMP
Markdown.parse("```julia\n" * read(joinpath(pkgdir(ReactiveMP), "docs", "nodes.jl"), String) * "\n```")
```

Declared, the node draws itself:

```@example example-node
using ReactiveMP # hide
include(joinpath(pkgdir(ReactiveMP), "docs", "nodes.jl"))
MessagePassingRulesBase.nodespec(Gaussian)
```
