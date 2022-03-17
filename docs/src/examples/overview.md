# [Examples overview](@id examples-overview)

This section contains a set of examples for Bayesian Inference with `ReactiveMP` package in various probabilistic models.

!!! note
    This section is WIP and more examples will be added over time. More examples can be found in [`demo/`](https://github.com/biaslab/ReactiveMP.jl/tree/master/demo) folder at GitHub repository.

- [Linear regression](@ref examples-linear-regression): An example of linear regression Bayesian inference.
- [Gaussian Linear Dynamical System](@ref examples-linear-gaussian-state-space-model): An example of inference procedure for Gaussian Linear Dynamical System with multivariate noisy observations using Belief Propagation (Sum Product) algorithm. Reference: [Simo Sarkka, Bayesian Filtering and Smoothing](https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf).
- [Hierarchical Gaussian Filter](@ref examples-hgf): An example of online inference procedure for Hierarchical Gaussian Filter with univariate noisy observations using Variational Message Passing algorithm. Reference: [Ismail Senoz, Online Message Passing-based Inference in the Hierarchical Gaussian Filter](https://ieeexplore.ieee.org/document/9173980).
- [Autoregressive Model](@ref examples-autoregressive): An example of variational Bayesian Inference on full graph for Autoregressive model. Reference: [Albert Podusenko, Message Passing-Based Inference for Time-Varying Autoregressive Models](https://www.mdpi.com/1099-4300/23/6/683).
- [Normalising Flows](@ref examples-flow): An example of variational Bayesian Inference with Normalizing Flows. Reference: Bard van Erp, Hybrid Inference with Invertible Neural Networks in Factor Graphs (submitted).
- [Gamma Mixture](@ref examples-gamma-mixture): This example implements one of the experiments outlined in https://biaslab.github.io/publication/mp-based-inference-in-gmm/ .