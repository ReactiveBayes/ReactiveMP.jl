# Getting started

`ReactiveMP.jl` is a Julia package for reactive message passing based Bayesian Inference on Factor Graphs. It supports both exact and variational inference.

`ReactiveMP.jl` package is a successor of the [`ForneyLab.jl`](https://github.com/biaslab/ForneyLab.jl) package. It follows the same ideas and concepts for message-passing based inference, but uses new reactive and efficient message passing implementation under the hood. The API between two packages is different due to a better flexibility, performance and new reactive approach for solving inference problems.

This page provides the necessary information you need to get started with `ReactiveMP.jl`. We will show the general approach to solving inference problems with `ReactiveMP.jl` by means of a running example: inferring the bias of a coin.

## Installation

Install `ReactiveMP.jl` through the Julia package manager:
```julia
] add https://github.com/biaslab/ReactiveMP.jl
```

!!! note
    For best user experience you also need to install `GraphPPL.jl`, `Rocket.jll` and `Distributions.jl` packages.

## Example: Inferring the bias of a coin
The `ReactiveMP.jl` approach to solving inference problems consists of three phases:

1. [Model specification](@ref): `ReactiveMP.jl` uses `GraphPPL.jl` package for model specification part. It offers a domain-specific language to specify your probabilistic model.
2. [Inference specification](@ref): `ReactiveMP.jl` does not restrict any certain use-cases for inference specification part. It has been designed to be as flexible as possible, but for most of the model it consists of the same simple building blocks. In this example we will show one of the many possible ways to infer your quantities of interest.
3. [Inference execution](@ref): Given model specification and inference procedure it is pretty straightforward to use reactive API from `Rocket.jl` to pass data to the inference backend and to run actual inference.

### Coin flip simulation
Let's start by creating some dataset. One approach could be flipping a coin N times and recording each outcome. Here, however, we will simulate this process reactively by sampling some values from a Bernoulli distribution infinitelly using `Rocket.jl` library. Each sample can be thought of as the outcome of single flip which is either heads or tails (1 or 0). We will assume that our virtual coin is biased, and lands heads up on 75% of the trials (on average).

```@example 1
using Rocket
using Distributions

n = 50
p = 0.75
distribution = Bernoulli(p)

stream = from(1:n) |> map(Int, (_) -> Int(rand(distribution)));
```