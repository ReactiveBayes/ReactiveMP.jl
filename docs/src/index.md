# ReactiveMP.jl

*A reactive message passing engine for Bayesian inference on factor graphs.*

ReactiveMP.jl runs message passing on a [factor graph](@extref MessagePassingRulesBase glossary-factor-graph).
It runs exact [belief propagation](@extref MessagePassingRulesBase glossary-belief-propagation),
[variational message passing](@extref MessagePassingRulesBase glossary-vmp) under a
[factorisation](@extref MessagePassingRulesBase glossary-factorisation), and the approximations
between them, with the [Bethe free energy](@extref MessagePassingRulesBase glossary-bethe-free-energy)
as the objective. It builds no schedule in advance. Messages and marginals are streams, and each
new observation propagates through the graph, recomputing only what depends on it. This suits
streaming and online inference as well as batch inference.

```@docs
ReactiveMP
```

## [Who this site is for](@id index-audience)

The engine is the computational core of the [RxInfer](https://reactivebayes.github.io/RxInfer.jl/stable/)
ecosystem. Most users never call it directly: RxInfer's `@model` and `infer` build the graph and
run it on this engine. This site is for you if you want to:

- build a factor graph by hand and run inference on it;
- write your own inference loop, or observe and change what the engine does, through callbacks,
  stream postprocessors and annotations;
- work on the engine itself.

If you want to specify and fit a model, start with
[RxInfer's documentation](https://reactivebayes.github.io/RxInfer.jl/stable/).

## [What the engine is and is not](@id index-scope)

The engine creates variables and factor nodes, connects them, and computes the messages, the
marginals and the free energy as data arrives.

The engine defines **no node** and **no rule**. A [factor node](@extref MessagePassingRulesBase glossary-factor-node)
and its [rules](@extref MessagePassingRulesBase glossary-rule) are declared with
[`MessagePassingRulesBase`](@extref MessagePassingRulesBase MessagePassingRulesBase), and the
nodes a model uses come from rule packages, such as
[`StandardMessagePassingRules`](@extref StandardMessagePassingRules StandardMessagePassingRules)
(see [The ecosystem](@ref ecosystem)). Loading a rule package is enough for the engine to find its
rules. This site's examples run on one small node, declared in the file that
[The example node](@ref example-node) shows.

The engine has **no model syntax**. You create each variable and node with a function call.
[GraphPPL](https://github.com/ReactiveBayes/GraphPPL.jl) turns a `@model` into a graph, and RxInfer
builds that graph on this engine.

## [Ideas and principles](@id index-ideas)

Reactive message passing creates no message passing schedule in advance. It reacts to changes in
the data, hence *reactive*. The PhD dissertation of Dmitry Bagaev,
[*Reactive Probabilistic Programming for Scalable Bayesian Inference*](https://pure.tue.nl/ws/portalfiles/portal/313860204/20231219_Bagaev_hf.pdf)
([also here](https://research.tue.nl/nl/publications/reactive-probabilistic-programming-for-scalable-bayesian-inferenc),
and [its sources](https://github.com/bvdmitri/phdthesis)), explains the ideas behind it. Tutorials
and example models are in the [RxInfer documentation](https://reactivebayes.github.io/RxInfer.jl/stable/).

## [The site](@id index-map)

- [Getting started](@ref getting-started) builds and runs a small model by hand, from the
  variables to the free energy.
- [The example node](@ref example-node) shows `Gaussian`, the node every example runs on, and
  its rules.
- **Concepts**: [factor graphs](@ref concepts-factor-graphs),
  [message passing](@ref concepts-message-passing), and the
  [inference lifecycle](@ref concepts-inference-lifecycle) every run goes through.
- **The engine**, in the order you build a graph: [variables](@ref lib-variables),
  [factor nodes](@ref lib-node), [activation options](@ref lib-activation-options),
  [messages](@ref lib-message), [marginals](@ref lib-marginal), [log scales](@ref lib-logscale),
  [form constraints](@ref custom-functional-form) and the [free energy](@ref lib-score).
- **Extension points**: [callbacks](@ref lib-callbacks) observe the engine,
  [stream postprocessors](@ref lib-stream-postprocessors) transform its streams, and
  [annotations](@ref lib-annotations) carry metadata on messages.
- [The ecosystem](@ref ecosystem) lists the rule packages, each with a site of its own.
- The [migration guides](@ref migration-v6-to-v7), [contributing](@ref contributing), and the
  [internals](@ref internals) a contributor needs.
