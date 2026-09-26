# [Messages](@id lib-message)

A message is what flows along an edge of the factor graph: a summary of the part of the graph it
comes from, about the variable on that edge. In belief propagation it is an unnormalised function,
in variational message passing the exponent of an expected log-factor (see
[Message passing](@ref concepts-message-passing)).

## [Messages as distributions](@id lib-messages-as-distributions)

A message is usually represented as a normalised probability distribution. A univariate normal
is two numbers, which is all that passes along the edge. The constant the normalisation leaves out
is the message's log scale, tracked when asked for (see [Log scales](@ref lib-logscale)).

```@example message
using ReactiveMP, BayesBase, ExponentialFamily, Distributions

message = Message(NormalMeanVariance(0.0, 1.0), false, false)
mean(message), var(message), logpdf(message, 1.0)
```

## [The message type](@id lib-message-type)

Every message is a [`Message`](@ref). It holds its data and forwards the statistics to it, and
records two flags: whether it is *clamped*, computed from constants and observations only, and
whether it is *initial*, set before inference or computed from initial values; the
[product of messages](@ref lib-messages-product) derives them from its sides'. It also carries its
log scale and an [`ReactiveMP.AnnotationDict`](@ref) of optional metadata (see
[Annotations](@ref lib-annotations)).

!!! note "Equality ignores annotations and log scales"
    `==` on `Message` and on [`Marginal`](@ref) compares the data and the two flags, not the
    annotations or the log scale, which describe *how* a value was computed rather than the belief
    it holds. Compare [`getannotations`](@ref) and [`getlogscale`](@ref) explicitly where they
    matter, such as in a cache key.

```@docs
AbstractMessage
Message
getdata(::Message)
is_clamped(::Message)
is_initial(::Message)
getannotations(::Message)
getlogscale(::Message)
as_message
```

## [Message streams](@id lib-message-observable)

Messages are not computed once and stored: each connection between a variable and a node carries
a *stream* of them, a [`ReactiveMP.MessageObservable`](@ref), which emits a new message whenever its
inputs change. The stream is lazy until activation connects it to its source; before that,
[`ReactiveMP.set_initial_message!`](@ref) can seed it, so that a rule that reads it at the start has
something to read. The latest message is kept: a subscriber that joins late receives it at once.

```@docs
ReactiveMP.MessageObservable
```

## [Product of messages](@id lib-messages-product)

A variable multiplies its inbound messages: all of them for its marginal, all but one for each
outbound message. The product of two messages is `BayesBase.prod` of their distributions, under
a product strategy, and is in general not normalised; its log scale accounts for the constant.
A [`ReactiveMP.MessageProductContext`](@ref) holds the settings: the product strategy, the form
constraint and when it applies (see [Form constraints](@ref custom-functional-form)), the order of
the fold, the annotation processors and the callbacks.

```@docs
ReactiveMP.MessageProductContext
ReactiveMP.compute_product_of_two_messages
ReactiveMP.compute_product_of_messages
ReactiveMP.MessagesProductFromLeftToRight
ReactiveMP.MessagesProductFromRightToLeft
```

## [Deferred messages](@id lib-messages-deferred)

A factor node emits its messages deferred: each is computed when a variable first reads it, and
cached, so that a message nobody reads is never computed.

```@docs
DeferredMessage
```

## [Message mappings](@id lib-messages-mapping)

A [`ReactiveMP.MessageMapping`](@ref) computes a node's message towards one interface: it
resolves the rule for the latest inputs, checks it, and runs it, or gives `missing` without
running a rule when an input is `missing`. It keeps the [scratch](@ref internals-scratch) of the
rule it runs between calls. The callback events of a rule call carry it, so it is what a callback
reads the node, the target and the algorithm from.

```@docs
ReactiveMP.MessageMapping
```

A mapping hands the rule the data of the messages and marginals it depends on, keyed as the rule
declares them, with their annotations alongside.

```@docs
ReactiveMP.rule_arguments
```
