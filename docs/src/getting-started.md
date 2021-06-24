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
using Rocket, GraphPPL, ReactiveMP, Distributions, Random
```

Next, lets define our dataset:

```@example coin
rng = MersenneTwister(42)
n = 10
p = 0.75
distribution = Bernoulli(p)

dataset = float.(rand(rng, Bernoulli(p), n))
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

```@example coin

# GraphPPL.jl export `@model` macro for model specification
# It accepts a regular Julia function and builds an FFG under the hood
@model function coin_model(n)

    # `datavar` creates data 'inputs' in our model
    # We will pass data later on to these inputs
    # In this example we create a sequence of inputs that accepts Float64
    y = datavar(Float64, n)
    
    # We endow θ parameter of our model with some prior
    θ ~ Beta(2.0, 7.0)
    
    # We assume that outcome of each coin flip is governed by the Bernoulli distribution
    for i in 1:n
        y[i] ~ Bernoulli(θ)
    end
    
    # We return references to our data inputs and θ parameter
    # We will use these references later on during inference step
    return y, θ
end

```

As you can see, `GraphPPL.jl` offers a model specification syntax that resembles closely to the mathematical equations defined above. We use `datavar` function to create "clamped" variables that take specific values at a later date. `θ ~ Beta(1.0, 1.0)` expression creates random variable `θ` and assigns it as an output of `Beta` node in the corresponding FFG. 

### Inference execution

Once we have defined our model, the next step is to use `ReactiveMP.jl` API to infer quantities of interests. To do this, we need to specify inference procedure. `ReactiveMP.jl` API is flexible in terms of inference specification and is compatible both with real-time inference processing and with statis datasets. In most of the cases for static datasets, as in our example, it consists of same basic building blocks:

1. Return variables of interests from model specification
2. Subscribe on variables of interests posterior marginal updates
3. Pass data to the model
4. Unsubscribe 

Here is an example of inference procedure:

```@example coin 
function inference(data)
    n = length(data)

    # `coin_model` function from `@model` macro returns a reference to the model object and 
    # the same output as in `return` statement in the original function specification
    model, (y, θ) = coin_model(n)
    
    # Reference for future posterior marginal 
    mθ = nothing

    # `getmarginal` function returns an observable of future posterior marginal updates
    # We use `Rocket.jl` API to subscribe on this observable
    # As soon as posterior marginal update is available we just save it in `mθ`
    subscription = subscribe!(getmarginal(θ), (m) -> mθ = m)
    
    # `update!` function passes data to our data inputs
    update!(y, data)
    
    # It is always a good practice to unsubscribe and to 
    # free computer resources held by the subscription
    unsubscribe!(subscription)
    
    # Here we return our resulting posterior marginal
    return mθ
end
```

```@example coin
θestimated = inference(dataset)
```

```@example coin
println("mean: ", mean(θestimated))
println("std:  ", std(θestimated))
nothing #hide
```

```@example coin
using Plots, LaTeXStrings; theme(:default)

rθ = range(0, 1, length = 1000)

p1 = plot(rθ, (x) -> pdf(Beta(2.0, 7.0), x), title="Prior", fillalpha=0.3, fillrange = 0, label=L"P\:(\theta)", c=1,)
p2 = plot(rθ, (x) -> pdf(θestimated, x), title="Posterior", fillalpha=0.3, fillrange = 0, label=L"P\:(\theta|y)", c=3)

plot(p1, p2, layout = @layout([ a; b ]))
```

In our dataset we used 10 coin flips to estimate the bias of a coin. It resulted in a vague posterior distribution, however `ReactiveMP.jl` scales very well for large models and factor graphs. We may use more coin flips in our dataset for better posterior distribution estimates:

```@example coin
dataset_100   = float.(rand(rng, Bernoulli(p), 100))
dataset_1000  = float.(rand(rng, Bernoulli(p), 1000))
dataset_10000 = float.(rand(rng, Bernoulli(p), 10000))
nothing # hide
```

```@example coin
θestimated_100   = inference(dataset_100)
θestimated_1000  = inference(dataset_1000)
θestimated_10000 = inference(dataset_10000)
nothing #hide
```

```@example coin
p3 = plot(title = "Posterior", legend = :topleft)

p3 = plot!(p3, rθ, (x) -> pdf(θestimated_100, x), fillalpha = 0.3, fillrange = 0, label = L"P\:(\theta\:|y_{1:100})", c = 4)
p3 = plot!(p3, rθ, (x) -> pdf(θestimated_1000, x), fillalpha = 0.3, fillrange = 0, label = L"P\:(\theta\:|y_{1:1000})", c = 5)
p3 = plot!(p3, rθ, (x) -> pdf(θestimated_10000, x), fillalpha = 0.3, fillrange = 0, label = L"P\:(\theta\:|y_{1:10000})", c = 6)

plot(p1, p3, layout = @layout([ a; b ]))
```

With larger dataset our posterior marginal estimate becomes more and more accurate and represents real value of the bias of a coin.

```@example coin
println("mean: ", mean(θestimated_10000))
println("std:  ", std(θestimated_10000))
nothing #hide
```

## Where to go next?
There are a set of [demos](https://github.com/biaslab/ReactiveMP.jl/tree/master/demo) available in `ReactiveMP.jl` repository that demonstrate the more advanced features of `ReactiveMP.jl`. Alternatively, you can head to the [User guide](@ref) which provides more detailed information of how to use `ReactiveMP.jl` to solve inference problems.