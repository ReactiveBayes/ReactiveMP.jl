
# [Messages implementation](@id lib-message)

In message passing framework one of the most important concepts is (wow!) messages. Messages flow on edges of a factor graph and usually hold some information in a form of probability distribution.
In ReactiveMP.jl we distinguish two major types of messages: Belief Propagation and Variational.  

## Abstract message type

Both belief propagation and variational messages are subtypes of a `AbstractMessage` supertype.

```@docs
AbstractMessage
```

## [Belief-Propagation (or Sum-Product) message](@id lib-belief-propagation-message)

Belief propagation messages are encoded with type `Message`. 

![message](../assets/img/bp-message.svg)
*Belief propagation message*

```@docs
Message
```

From implementation point a view `Message` structure does nothing but holds some `data` object and redirects most of the statistical related functions to that `data` object. However it used extensively in Julia's multiple dispatch. Implementation also uses extra `is_initial` and `is_clamped` fields to determine if product of two messages results in `is_initial` or `is_clamped` posterior marginal.

```@setup bp-message
using Rocket, GraphPPL, ReactiveMP, Distributions, Random
```

```@example bp-message
distribution = NormalMeanPrecision(0.0, 1.0)
message      = Message(distribution, false, true, nothing)
```

```@example bp-message
mean(message), precision(message)
```

```@example bp-message
logpdf(message, 1.0)
```

```@example bp-message
is_clamped(message), is_initial(message)
```

The user should not really interact with `Message` structure while working with `ReactiveMP` unless doing some advanced inference procedures that involve prediction.

## [Variational message](@id lib-variational-message)

Variational messages are encoded with type `VariationalMessage`.

![message](../assets/img/vmp-message.svg)
*Variational message with structured factorisation q(x, y)q(z) assumption*

