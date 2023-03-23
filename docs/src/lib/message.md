
# [Messages implementation](@id lib-message)

In our message passing framework one of the most important concepts is the message (wow!).
Messages flow along edges of a factor graph and hold information about the part of the graph that it originates from.
Usually this information is in a form of a probability distribution.
Two common messages are belief propagation messages and variational messages, with are computed differently as shown below.

## Abstract message type

Both belief propagation and variational messages are subtypes of a `AbstractMessage` supertype.

```@docs
AbstractMessage
```

## [Belief-Propagation (or Sum-Product) message](@id lib-belief-propagation-message)

![message](../assets/img/bp-message.svg)
*Belief propagation message*

## [Variational message](@id lib-variational-message)

![message](../assets/img/vmp-message.svg)
*Variational message with structured factorisation q(x, y)q(z) assumption*

## Message type

All messages are encoded with the type `Message`. 

```@docs
Message
ReactiveMP.materialize!
```

From an implementation point a view the `Message` structure does nothing but hold some `data` object and redirects most of the statistical related functions to that `data` object. 
However, this object is used extensively in Julia's multiple dispatch. 
Our implementation also uses extra `is_initial` and `is_clamped` fields to determine if product of two messages results in `is_initial` or `is_clamped` posterior marginal.
The final field contains the addons. These contain additional information on top of the functional form of the distribution, such as its scaling or computation history.

```@setup bp-message
using ReactiveMP
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

