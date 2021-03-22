# ReactiveMP.jl

ReactiveMP.jl is a Julia package for automatic Bayesian inference on a factor graph with reactive message passing.

The current version supports belief propagation (sum-product message passing) and variational message passing (both Mean-Field and Structured VMP).

# Development

The framework is in active development stage and should not be considered as production ready. The core and API may be changed in the near future.

# Getting Started

There are demos available to get you started in the `demo/` folder. Comparative benchmarks are available in the `benchmarks/` folder.

## ReactiveMP.jl is fast

ReactiveMP.jl has been designed with a focus on efficiency, scalability and maximum performance for running inference on conjugate state-space models. Below is a benchmark comparison between ReactiveMP.jl and Turing.jl on a linear multivariate Gaussian state space Model. It is worth noting that this model contains many conjugate prior and likelihood pairings that lead to analytically computable Bayesian posteriors. For these types of models, ReactiveMP.jl takes advantage of the conjugate pairings and beats general-purpose probabilistic programming packages like Turing.jl easily in terms of computational load, speed, memory  and accuracy. On the other hand, Turing.jl is currently still capable of running inference for a broader set of models. 

Code is available in [benchmarks folder](https://github.com/biaslab/ReactiveMP.jl/tree/master/benchmarks):

![ReactiveMP.jl Benchmark](benchmarks/plots/lgssm.svg?raw=true&sanitize=true "ReactiveMP.jl Benchmark")


