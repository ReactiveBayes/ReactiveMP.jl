# ReactiveMP.jl

ReactiveMP.jl is a Julia package for automatic Bayesian inference on a factor graph with reactive message passing.

The current version supports belief propagation (sum-product message passing) and variational message passing (both Mean-Field and Structured VMP).

# Getting Started

There are demos available to get you started in the `demo/` folder. Comparative benchmarks are available in the `benchmarks/` folder.

# Development

The framework is in active development stage. API may change in the near future.

## ReactiveMP.jl is fast

ReactiveMP.jl has been designed with a focus on efficiency, scalability and maximum performance. Below is a benchmark comparison between Rocket.jl and [Turing.jl](https://github.com/TuringLang/Turing.jl) on a Linear Multivariate Gaussian State Space Model. It is worth to note that this model is a conjugate model, hence message passing has a big advantage over general-purpose probabilistic programming packages like Turing.jl. However it is important to have different solutions for different problems and ReactiveMP.jl is better suitable for conjugate models, uses less memory and provides much better performance. While being faster ReactiveMP.jl also gives more accurate inference estimates for model hidden states and parameters since message passing uses analytical rules for marginalisation and integrals evaluation.

Code is available in [benchmarks folder](https://github.com/biaslab/ReactiveMP.jl/tree/master/benchmarks):

![ReactiveMP.jl Benchmark](benchmarks/plots/lgssm.svg?raw=true&sanitize=true "ReactiveMP.jl Benchmark")


