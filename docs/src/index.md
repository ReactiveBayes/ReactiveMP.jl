ReactiveMP.jl
=============

*Julia package for reactive message passing Bayesian inference engine on a factor graph.*

!!! note
    This package exports only an inference engine, for the full ecosystem with convenient model and constraints specification we refer user to the [`RxInfer.jl`](https://github.com/reactivebayes/RxInfer.jl) package and its [documentation](https://reactivebayes.github.io/RxInfer.jl/stable/).

## Ideas and principles behind `ReactiveMP.jl`

`ReactiveMP.jl` is a particular implementation of message passing on factor graphs, which does not create any specific message passing schedule in advance, but rather _reacts_ on changes in the data source (hence _reactive_ in the name of the package). The detailed explanation of the ideas and principles behind the _Reactive Message Passing_ can be found in PhD disseration of _Dmitry Bagaev_ titled [__Reactive Probabilistic Programming for Scalable Bayesian Inference__](https://pure.tue.nl/ws/portalfiles/portal/313860204/20231219_Bagaev_hf.pdf) ([link2](https://research.tue.nl/nl/publications/reactive-probabilistic-programming-for-scalable-bayesian-inferenc), [link3](https://github.com/bvdmitri/phdthesis)).

## Examples and tutorials

The `ReactiveMP.jl` package is intended for advanced users with a deep understanding of message passing principles. For accesible tutorials and examples are available in the [RxInfer documentation](https://reactivebayes.github.io/RxInfer.jl/stable/).

## Table of Contents

```@contents
Pages = [
  "lib/message.md",
  "lib/node.md",
  "lib/math.md",
  "lib/rules/rules.md",
  "lib/nodes/nodes.md",
  "extra/contributing.md",
  "extra/extensions.md"
]
Depth = 2
```

## Index

```@index
```
