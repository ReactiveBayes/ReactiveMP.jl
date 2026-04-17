ReactiveMP.jl
=============

*Julia package for reactive message passing Bayesian inference engine on a factor graph.*

`ReactiveMP.jl` is a low-level inference engine that implements variational message passing on factor graphs. It is designed for advanced users who need fine-grained control over message passing, custom factor nodes, and custom update rules. For most use cases, the [RxInfer.jl](https://github.com/reactivebayes/RxInfer.jl) package provides a convenient model specification layer on top of ReactiveMP.jl.

!!! note
    This package exports only an inference engine. For the full ecosystem with convenient model and constraints specification, see [`RxInfer.jl`](https://github.com/reactivebayes/RxInfer.jl) and its [documentation](https://reactivebayes.github.io/RxInfer.jl/stable/).

## [Start here](@id index-start-here)

If you are new to ReactiveMP.jl, read the Concepts section first. It explains the key ideas without assuming prior familiarity with the codebase:

1. **[Factor graphs](@ref concepts-factor-graphs)** — what factor graphs are and how ReactiveMP.jl represents them.
2. **[Message passing](@ref concepts-message-passing)** — how belief propagation and variational message passing work, and the reactive computation model.
3. **[Inference lifecycle](@ref concepts-inference-lifecycle)** — the three phases every inference run goes through: construction, activation, and observation.

After reading the Concepts section, the Library section provides the full API reference for each component.

## [Ideas and principles behind `ReactiveMP.jl`](@id index-ideas)

`ReactiveMP.jl` is a particular implementation of message passing on factor graphs, which does not create any specific message passing schedule in advance, but rather _reacts_ on changes in the data source (hence _reactive_ in the name of the package). The detailed explanation of the ideas and principles behind the _Reactive Message Passing_ can be found in PhD dissertation of _Dmitry Bagaev_ titled [__Reactive Probabilistic Programming for Scalable Bayesian Inference__](https://pure.tue.nl/ws/portalfiles/portal/313860204/20231219_Bagaev_hf.pdf) ([link2](https://research.tue.nl/nl/publications/reactive-probabilistic-programming-for-scalable-bayesian-inferenc), [link3](https://github.com/bvdmitri/phdthesis)).

## [Examples and tutorials](@id index-examples)

The `ReactiveMP.jl` package is intended for advanced users with a deep understanding of message passing principles. 
Accessible tutorials and examples are available in the [RxInfer documentation](https://reactivebayes.github.io/RxInfer.jl/stable/).

## Table of Contents

```@contents
Pages = [
  "concepts/factor-graphs.md",
  "concepts/message-passing.md",
  "concepts/inference-lifecycle.md",
  "lib/nodes.md",
  "lib/variables.md",
  "lib/message.md",
  "lib/marginal.md",
  "lib/rules.md",
  "lib/helpers.md",
  "lib/algebra.md",
  "extra/contributing.md",
  "extra/extensions.md",
  "extra/methods.md",
]
Depth = 2
```

## Index

```@index
```
