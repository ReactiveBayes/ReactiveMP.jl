<img src="docs/src/assets/logo.svg?raw=true&sanitize=true" width="50%">

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://reactivebayes.github.io/ReactiveMP.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://reactivebayes.github.io/ReactiveMP.jl/dev)
[![Build Status](https://github.com/reactivebayes/ReactiveMP.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/reactivebayes/ReactiveMP.jl/actions)
[![Coverage](https://codecov.io/gh/reactivebayes/ReactiveMP.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/reactivebayes/ReactiveMP.jl)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.8381133.svg)](https://zenodo.org/doi/10.5281/zenodo.5913616)

# Overview

`ReactiveMP.jl` is a Julia package that provides an efficient reactive message passing based Bayesian inference engine on a factor graph. The package is a part of the bigger and user-friendly ecosystem for automatic Bayesian inference called [RxInfer](https://github.com/reactivebayes/RxInfer.jl). While ReactiveMP.jl exports only the inference engine, RxInfer provides convenient tools for model and inference constraints specification as well as routines for running efficient inference both for static and dynamic datasets.

ReactiveMP.jl is designed for advanced users who need fine-grained control over message passing, custom factor nodes, and custom update rules. It does not create a specific message passing schedule in advance, but rather _reacts_ on changes in the data source (hence _reactive_ in the name of the package).

# Installation

Install ReactiveMP through the Julia package manager:

```julia
] add ReactiveMP
```

Optionally, use `] test ReactiveMP` to validate the installation by running the test suite.

# Documentation

For more information about `ReactiveMP.jl` please refer to the [documentation](https://reactivebayes.github.io/ReactiveMP.jl/stable).

# Examples and tutorials

The ReactiveMP.jl package is intended for advanced users with a deep understanding of message passing principles. Accessible tutorials and examples are available in the [RxInfer documentation](https://reactivebayes.github.io/RxInfer.jl/stable/).

# Ecosystem

The `RxInfer` framework consists of four *core* packages developed by [ReactiveBayes](https://github.com/reactivebayes/):

- [`ReactiveMP.jl`](https://github.com/reactivebayes/ReactiveMP.jl) - the underlying message passing-based inference engine (this package)
- [`RxInfer.jl`](https://github.com/reactivebayes/RxInfer.jl) - user-friendly modeling and inference layer
- [`GraphPPL.jl`](https://github.com/reactivebayes/GraphPPL.jl) - model and constraints specification package
- [`ExponentialFamily.jl`](https://github.com/reactivebayes/ExponentialFamily.jl) - package for exponential family distributions
- [`Rocket.jl`](https://github.com/reactivebayes/Rocket.jl) - reactive extensions package for Julia

# References

- [A Julia package for reactive variational Bayesian inference](https://doi.org/10.1016/j.simpa.2022.100299) - a reference paper for the `ReactiveMP.jl` package.
- [Reactive Probabilistic Programming for Scalable Bayesian Inference](https://pure.tue.nl/ws/portalfiles/portal/313860204/20231219_Bagaev_hf.pdf) - a PhD dissertation outlining core ideas and principles behind ReactiveMP ([link2](https://research.tue.nl/nl/publications/reactive-probabilistic-programming-for-scalable-bayesian-inferenc), [link3](https://github.com/bvdmitri/phdthesis)).
- [Variational Message Passing and Local Constraint Manipulation in Factor Graphs](https://doi.org/10.3390/e23070807) - describes theoretical aspects of the underlying Bayesian inference method.
- [Reactive Message Passing for Scalable Bayesian Inference](https://doi.org/10.48550/arXiv.2112.13251) - describes implementation aspects of the Bayesian inference engine and performs benchmarks and accuracy comparison on various models.

# License

MIT License Copyright (c) 2021-2024 BIASlab, 2024-present ReactiveBayes
