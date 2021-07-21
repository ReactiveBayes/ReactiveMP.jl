
# [Messages implementation](@id lib-message)

In message passing framework one of the most important concept is (wow!) messages. Messages flow on edges of a factor graph and usually hold some information in a form of probability distribution.
In ReactiveMP.jl we distinguish two major types of messages: Belief Propagation and Variational.  

## Abstract message type

Both belief propagation and variational messages are subtypes of a `AbstractMessage` supertype.

```@docs
AbstractMessage
```

## Belief propagation message

![message](../assets/img/message.svg)

```@docs
Message
```