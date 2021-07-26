ReacitveMP.jl
=============

*Julia package for automatic Bayesian inference on a factor graph with reactive message passing.*

Given a probabilistic model, ReactiveMP allows for an efficient message-passing based Bayesian inference. It uses the model structure to generate an algorithm that consists of a sequence of local computations on a Forney-style factor graph (FFG) representation of the model.

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
Head to the [Getting started](@ref user-guide-getting-started) section to get up and running with ForneyLab.

## Table of Contents

```@contents
Pages = [
  "man/getting-started.md",
  "man/model-specification.md",
  "lib/message.md",
  "lib/node.md",
  "extra/contributing.md"
]
Depth = 2
```

## Index

```@index
```
