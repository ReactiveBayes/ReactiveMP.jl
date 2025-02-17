# ReactiveMP.jl

| **Documentation**                                                         | **Build Status**                 | **Coverage**                       | **Zenodo DOI**                   |
|:-------------------------------------------------------------------------:|:--------------------------------:|:----------------------------------:|:--------------------------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![CI][ci-img]][ci-url]         | [![Codecov][codecov-img]][codecov-url] | [![DOI][zenodo-img]][zenodo-url] |

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://reactivebayes.github.io/ReactiveMP.jl/dev

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://reactivebayes.github.io/ReactiveMP.jl/stable

[ci-img]: https://github.com/reactivebayes/ReactiveMP.jl/actions/workflows/ci.yml/badge.svg?branch=main
[ci-url]: https://github.com/reactivebayes/ReactiveMP.jl/actions

[codecov-img]: https://codecov.io/gh/reactivebayes/ReactiveMP.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/reactivebayes/ReactiveMP.jl?branch=main

[zenodo-img]: https://zenodo.org/badge/DOI/10.5281/zenodo.8381133.svg
[zenodo-url]: https://zenodo.org/doi/10.5281/zenodo.5913616

# Reactive message passing engine

ReactiveMP.jl is a Julia package that provides an efficient reactive message passing based Bayesian inference engine on a factor graph. The package is a part of the bigger and user-friendly ecosystem for automatic Bayesian inference called [RxInfer](https://github.com/reactivebayes/RxInfer.jl). While ReactiveMP.jl exports only the inference engine, RxInfer provides convenient tools for model and inference constraints specification as well as routines for running efficient inference both for static and dynamic datasets. 

## Examples and tutorials

The ReactiveMP.jl package is intended for advanced users with a deep understanding of message passing principles.
Accesible tutorials and examples are available in the [RxInfer documentation](https://reactivebayes.github.io/RxInfer.jl/stable/).

# License

MIT License Copyright (c) 2021-2024 BIASlab, 2024-present ReactiveBayes
