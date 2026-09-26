# ReactiveMP.jl

*A reactive message passing engine for Bayesian inference on factor graphs.*

ReactiveMP.jl runs message passing on a factor graph: exact belief propagation, variational
message passing under a factorisation, and the approximations between them, with the Bethe free
energy as the objective. It builds no schedule in advance. Messages and marginals are streams,
and each new observation propagates through the graph, recomputing only what depends on it; this
suits streaming and online inference as well as batch inference.

The engine is the computational core of the [RxInfer](https://github.com/ReactiveBayes/RxInfer.jl)
ecosystem, and is not usually used directly: RxInfer's `@model` and `infer` build the graph and run
it on this engine. Use ReactiveMP.jl directly to build graphs by hand, to write your own
inference loop, or to work on the engine itself. The nodes and their update rules are not part of
the engine: they come from rule packages (see [The ecosystem](@ref ecosystem)).

```@docs
ReactiveMP
```

## [Ideas and principles](@id index-ideas)

Reactive message passing does not create a message passing schedule in advance, but reacts to
changes in the data (hence *reactive*). The ideas behind it are explained in the PhD
dissertation of Dmitry Bagaev,
[*Reactive Probabilistic Programming for Scalable Bayesian Inference*](https://pure.tue.nl/ws/portalfiles/portal/313860204/20231219_Bagaev_hf.pdf)
([also here](https://research.tue.nl/nl/publications/reactive-probabilistic-programming-for-scalable-bayesian-inferenc),
and [its sources](https://github.com/bvdmitri/phdthesis)). Tutorials and examples of models are in
the [RxInfer documentation](https://reactivebayes.github.io/RxInfer.jl/stable/).

## [The site](@id index-map)

- [Getting started](@ref getting-started) builds and runs a small model by hand, from the
  variables to the free energy.
- **Concepts**: [factor graphs](@ref concepts-factor-graphs),
  [message passing](@ref concepts-message-passing), and the
  [inference lifecycle](@ref concepts-inference-lifecycle) every run goes through.
- **The engine**, in the order a graph is built: [variables](@ref lib-variables),
  [factor nodes](@ref lib-node), [activation options](@ref lib-activation-options),
  [messages](@ref lib-message), [marginals](@ref lib-marginal), [log scales](@ref lib-logscale),
  [form constraints](@ref custom-functional-form) and the [free energy](@ref lib-score).
- **Extension points**: [callbacks](@ref lib-callbacks), which observe the engine,
  [stream postprocessors](@ref lib-stream-postprocessors), which transform its streams, and
  [annotations](@ref lib-annotations), which carry metadata on messages.
- [The ecosystem](@ref ecosystem) lists the rule packages, each with a site of its own.
- The [migration guides](@ref migration-v6-to-v7), [contributing](@ref contributing), and the
  [internals](@ref internals) a contributor needs.
