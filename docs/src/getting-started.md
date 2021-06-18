# Getting started

`ReactiveMP.jl` is a Julia package for Bayesian Inference on Factor Graphs by Message Passing. It supports both exact and variational inference algorithms.

`ReactiveMP.jl` package is a successor of the [`ForneyLab.jl`](https://github.com/biaslab/ForneyLab.jl) package. It follows the same ideas and concepts for message-passing based inference, but uses new reactive and efficient message passing implementation under the hood. The API between two packages is different due to a better flexibility, performance and new reactive approach for solving inference problems.

This page provides the necessary information you need to get started with `ReactiveMP.jl`. We will show the general approach to solving inference problems with `ReactiveMP.jl` by means of a running example: inferring the bias of a coin.

## Installation

Install `ReactiveMP.jl` through the Julia package manager:
```julia
] add ReactiveMP
```

!!! note
    For best user experience you also need to install `GraphPPL.jl`, `Rocket.jl` and `Distributions.jl` packages.

## Example: Inferring the bias of a coin
The `ReactiveMP.jl` approach to solving inference problems consists of three phases:

1. [Model specification](@ref): `ReactiveMP.jl` uses `GraphPPL.jl` package for model specification part. It offers a domain-specific language to specify your probabilistic model.
2. [Inference specification](@ref): `ReactiveMP.jl` inference API has been designed to be as flexible as possible and it is compatible both with asynchronous infinite data streams and with static datasets. For most of the use cases it consists of the same simple building blocks. In this example we will show one of the many possible ways to infer your quantities of interest.
3. [Inference execution](@ref): Given model specification and inference procedure it is pretty straightforward to use reactive API from `Rocket.jl` to pass data to the inference backend and to run actual inference.

### Coin flip simulation
Let's start by creating some dataset. One approach could be flipping a coin N times and recording each outcome. Here, however, we will simulate this process by sampling some values from a Bernoulli distribution using streams from `Rocket.jl` library. For simplicity in this example we will use static pre-generated dataset. Each sample can be thought of as the outcome of single flip which is either heads or tails (1 or 0). We will assume that our virtual coin is biased, and lands heads up on 75% of the trials (on average).

First lets setup our environment by importing all needed packages:

```@example coin
using Rocket, GraphPPL, ReactiveMP, Distributions
```

Next, lets define our dataset:

```@example coin
n = 50
p = 0.75
distribution = Bernoulli(p)

stream = from(1:n) |> map(Int, (_) -> Int(rand(distribution)))

nothing # hide
```

We use `from(...)` function from `Rocket.jl` that generates an observable from given iterable sequence `1:n`. We then transform this sequence with the help of the "pipe" syntax and the `map` operator. For each value in our stream we return a random sample from `Bernoulli(p)` distribution.

```@example coin
# Here we use `subscribe!` function from `Rocket.jl`
# We take first 15 emissions from `stream` and reduce it into an array with `to_array` operator
subscription = subscribe!(stream |> take(15) |> to_array(), (v) -> println("dataset[1:15] = ", v))
nothing # hide
```

It is always a good practice to `unsubscribe!` every time to release computer resources held by the subscription especially when working with asynchronous data streams.

```@example coin
unsubscribe!(subscription)
```

### Model specification

In a Bayesian setting, the next step is to specify our probabilistic model. This amounts to specifying the joint probability of the random variables of the system.

#### Likelihood
We will assume that the outcome of each coin flip is governed by the Bernoulli distribution, i.e.

```math 
y_i \sim \mathrm{Bernoulli}(\theta),
```

where ``y_i = 1`` represents "heads", ``y_i = 0`` represents "tails". The underlying probability of the coin landing heads up for a single coin flip is ``\theta \in [0,1]``.

#### Prior
We will choose the conjugate prior of the Bernoulli likelihood function defined above, namely the beta distribution, i.e.

```math 
\theta \sim Beta(a, b),
```

where ``a`` and ``b`` are the hyperparameters that encode our prior beliefs about the possible values of ``\theta``. We will assign values to the hyperparameters in a later step.   

#### Joint probability
The joint probability is given by the multiplication of the likelihood and the prior, i.e.

```math
P(y_{1:N}, θ) = P(θ) \prod_{i=1}^N P(y_i | θ).
```

Now let's see how to specify this model using GraphPPL's package syntax.