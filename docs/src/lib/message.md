
# [Messages implementation](@id lib-message)

In the message passing framework, one of the most important concepts is the message.
Given a factor graph, messages are arbitrary functions that flow along the edges of the graph and hold information about the part of the graph from which they originate.

## [Message as a distribution](@id lib-messages-as-distributions)

Often, a message can be represented in the form of a probability distribution, as a probability distribution can be matched with its _probability density function_.
The representation of messages as probability distributions is not only for convenience but also for performance reasons. For example, a univariate _Gaussian_ distribution can be parameterized with two numbers, which significantly reduce the amount of information needed to pass along the edges of the graph.

```@example
using StatsPlots, Distributions

plot(Normal(0.0, 1.0), label = "Univariate Gaussian distribution", fillalpha = 0.4, fill = 0)
```

## [Variational Message Passing](@id lib-message-vmp)

The message passing technique is useful for finding the posterior distribution over certain parameters in a model, originating from exact Bayesian inference, which is also known as __Belief Propagation__. However, the message passing technique can also be used to find _approximate_ solutions to posteriors - a technique known as __Variational inference__. The `ReactiveMP.jl` package implements __Variational Message Passing__ since it is a more general form than exact inference, and also because the exact solution can be framed as an approximate solution subject to no constraints. Here are visual schematics of the differences between messages in Belief propagation and Variational inference.

### [Belief-Propagation (or Sum-Product) message](@id lib-belief-propagation-message)

![message](../assets/img/bp-message.svg)
*Belief propagation message*

### [Variational message](@id lib-variational-message)

![message](../assets/img/vmp-message.svg)
*Variational message with structured factorisation q(x, y)q(z) assumption*


## Message type

All messages are encoded with the type `Message`. 

```@docs
AbstractMessage
Message
```

From an implementation point a view the `Message` structure does nothing but hold some `data` object and redirects most of the statistical related functions to that `data` object. 
However, this object is used extensively in Julia's multiple dispatch. 
Our implementation also uses extra `is_initial` and `is_clamped` fields to determine if [product of two messages](@ref lib-messages-product) results in `is_initial` or `is_clamped` posterior marginal. The final field contains the addons. These contain additional information on top of the functional form of the distribution, such as its scaling or computation history.

```@docs
ReactiveMP.getdata(message::Message)
ReactiveMP.is_clamped(message::Message)
ReactiveMP.is_initial(message::Message)
ReactiveMP.getaddons(message::Message)
ReactiveMP.as_message
```

```@example message
using ReactiveMP, BayesBase, ExponentialFamily

distribution = ExponentialFamily.NormalMeanPrecision(0.0, 1.0)
message      = Message(distribution, false, true, nothing)
```

```@example message
mean(message), precision(message)
```

```@example message
logpdf(message, 1.0)
```

```@example message
is_clamped(message), is_initial(message)
```

### [Product of messages](@id lib-messages-product)

In message passing framework, in order to compute a posterior we must compute a normalized product of two messages.
For this purpose the `ReactiveMP.jl` uses the `multiply_messages` function, which internally uses the `prod` function
defined in `BayesBase.jl` with various product strategies. We refer an interested reader to the documentation of the 
`BayesBase.jl` package for more information.

```@docs
ReactiveMP.multiply_messages
ReactiveMP.messages_prod_fn
```


### [Deferred messages](@id lib-messages-deferred)

```@docs 
ReactiveMP.DeferredMessage
```