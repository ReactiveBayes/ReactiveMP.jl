ReactiveMP.jl
=============

*Julia package for automatic Bayesian inference on a factor graph with reactive message passing.*

Given a probabilistic model, ReactiveMP allows for an efficient message-passing based Bayesian inference. It uses the model structure to generate an algorithm that consists of a sequence of local computations on a Forney-style factor graph (FFG) representation of the model.

ReactiveMP.jl has been designed with a focus on efficiency, scalability and maximum performance for running inference with message passing. It is worth noting that this package is aimed to run Bayesian inference for conjugate state-space models. For these types of models, ReactiveMP.jl takes advantage of the conjugate pairings and beats general-purpose probabilistic programming packages easily in terms of computational load, speed, memory  and accuracy. On the other hand, sampling-based packages like [Turing.jl](https://github.com/TuringLang/Turing.jl) are generic Bayesian inference solutions and are capable of running inference for a broader set of models. 

## Package Features

- User friendly syntax for specification of probabilistic models.
- Automatic generation of message passing algorithms including
    - [Belief propagation](https://en.wikipedia.org/wiki/Belief_propagation)
    - [Variational message passing](https://en.wikipedia.org/wiki/Variational_message_passing)
    - [Expectation maximization](https://en.wikipedia.org/wiki/Expectation-maximization_algorithm)
- Support for hybrid models combining discrete and continuous latent variables.
- Support for hybrid distinct message passing inference algorithm under a unified paradigm.
- Evaluation of Bethe free energy as a model performance measure.
- Schedule-free reactive message passing API.
- High performance.
- Scalability for large models with millions of parameters and observations.
- Inference procedure is differentiable.
- Easy to extend with custom nodes.

## Resources

- For an introduction to message passing and FFGs, see [The Factor Graph Approach to Model-Based Signal Processing](https://ieeexplore.ieee.org/document/4282128/) by Loeliger et al. (2007).

## How to get started?
Head to the [Getting started](@ref user-guide-getting-started) section to get up and running with ForneyLab. Alternatively, explore various [examples](@ref examples-overview) in the documentation. For advanced extensive tutorial take a look on [Advanced Tutorial](@ref user-guide-advanced-tutorial).

## Table of Contents

```@contents
Pages = [
  "man/getting-started.md",
  "man/advanced-tutorial.md",
  "man/model-specification.md",
  "man/constraints-specification.md",
  "man/meta-specification.md",
  "examples/overview.md",
  "lib/message.md",
  "lib/node.md",
  "lib/math.md",
  "extra/contributing.md"
]
Depth = 2
```

## Index

```@index
```
