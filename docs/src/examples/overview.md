# [Examples overview](@id examples-overview)

This section contains a set of examples for Bayesian Inference with `ReactiveMP` package in various probabilistic models.

!!! note
    This section is WIP and more examples will be added over time. More examples can be found in [`demo/`](https://github.com/biaslab/ReactiveMP.jl/tree/master/demo) folder at GitHub repository.

- [Gaussian Linear Dynamical System](@ref examples-linear-gaussian-state-space-model): An example of inference procedure for Gaussian Linear Dynamical System with multivariate noisy observations using Belief Propagation (Sum Product) algorithm. Reference: [Simo Sarkka, Bayesian Filtering and Smoothing](https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf).
- [Hierarchical Gaussian Filter](@ref example-hgf): An example of online inference procedure for Hierarchical Gaussian Filter wiht univariate noisy observations using Variational Message Passing algorithm. Reference: [Ismail Senoz, Online Message Passing-based Inference in the Hierarchical Gaussian Filter](https://ieeexplore.ieee.org/document/9173980)