# [Messages](@id lib-message)

A [message](@extref MessagePassingRulesBase glossary-message) is what flows along an edge of the
factor graph: a summary, about the variable on that edge, of the part of the graph it comes from.
In belief propagation it is an unnormalised function. In variational message passing it is the
exponent of an expected log-factor (see [Message passing](@ref concepts-message-passing)).

```@setup message
using ReactiveMP, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, Rocket
import ReactiveMP: activate!, FactorNodeActivationOptions, MessageProductContext, get_stream_of_marginals
include(joinpath(pkgdir(ReactiveMP), "docs", "nodes.jl"))
```

The examples use the `Gaussian` node of [The example node](@ref example-node).

## [Messages as distributions](@id lib-messages-as-distributions)

A message is usually a normalised probability distribution. A univariate normal is two numbers,
which is all that passes along the edge:

```@example message
message = Message(NormalMeanVariance(0.0, 1.0), false, false)
mean(message), var(message), logpdf(message, 1.0)
```

The message forwards the statistics to its distribution. The constant the normalisation leaves
out is the message's log scale, which the engine tracks when you ask it to (see
[Log scales](@ref lib-logscale)).

## [The message type](@id lib-message-type)

Every message is a [`Message`](@ref). It holds its data and records two flags:

- *clamped*: the message is computed from constants and observations only;
- *initial*: the message was set before inference, or computed from initial values.

The [product of messages](@ref lib-messages-product) derives both flags from its sides'. A
message also carries its log scale and a [`ReactiveMP.AnnotationDict`](@ref) of optional
metadata (see [Annotations](@ref lib-annotations)).

```@example message
tracked = Message(NormalMeanVariance(0.0, 1.0), false, false, ReactiveMP.AnnotationDict(), -1.5)
tracked == message, getlogscale(tracked)
```

!!! note "Equality ignores annotations and log scales"
    `==` on `Message` and on [`Marginal`](@ref) compares the data and the two flags, not the
    annotations or the log scale. These describe *how* a value was computed, not the belief it
    holds. Compare [`getannotations`](@ref) and [`getlogscale`](@ref) explicitly where they
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

The engine does not compute a message once and store it. Each connection between a variable and
a node carries a *stream* of messages, a [`ReactiveMP.MessageObservable`](@ref), which emits a new
message whenever its inputs change. The stream is lazy until activation connects it to its
source. Before that, [`ReactiveMP.set_initial_message!`](@ref) can seed it, so that a rule that
reads it at the start has something to read. The stream keeps its latest message: a subscriber
that joins late receives it at once.

```@docs
ReactiveMP.MessageObservable
```

## [Product of messages](@id lib-messages-product)

A variable multiplies its inbound messages: all of them for its marginal, and all but one for
each outbound message. The product of two messages is `BayesBase.prod` of their distributions,
under a product strategy. It is in general not normalised, and its log scale accounts for the
constant.

A [`ReactiveMP.MessageProductContext`](@ref) holds the settings: the product strategy, the
[form constraint](@ref custom-functional-form) and when it applies, the order of the fold, the
annotation processors and the callbacks.

```@example message
messages = [
    Message(NormalMeanVariance(0.0, 10.0), true, false),
    Message(NormalMeanVariance(2.0, 1.0), false, true),
]
product = ReactiveMP.compute_product_of_messages(randomvar(), MessageProductContext(), messages)
getdata(product), is_clamped(product), is_initial(product)
```

One side is clamped and the other initial, so the product is initial.

```@docs
ReactiveMP.MessageProductContext
ReactiveMP.compute_product_of_two_messages
ReactiveMP.compute_product_of_messages
ReactiveMP.MessagesProductFromLeftToRight
ReactiveMP.MessagesProductFromRightToLeft
```

## [Deferred messages](@id lib-messages-deferred)

A factor node emits its messages deferred. The engine computes each when a variable first reads
it, and caches it, so that a message nobody reads is never computed. [`as_message`](@ref)
computes a deferred message.

```@docs
DeferredMessage
```

## [Message mappings](@id lib-messages-mapping)

A [`ReactiveMP.MessageMapping`](@ref) computes a node's message towards one interface. It
resolves the rule for the latest inputs, checks it, and runs it. When an input is `missing`, it
gives `missing` without running a rule. It keeps the [scratch](@ref internals-scratch) of the
rule it runs between calls. The callback events of a rule call carry it, so a callback reads the
node, the target and the algorithm from it:

```@example message
x, y = randomvar(label = :x), datavar(label = :y)
prior = factornode(Gaussian, [(:out, x), (:μ, constvar(0.0)), (:v, constvar(10.0))])
likelihood = factornode(Gaussian, [(:out, y), (:μ, x), (:v, constvar(1.0))])

mappings = []
callbacks = (before_message_rule_call = (event) -> push!(mappings, event.mapping),)
activate!(x, RandomVariableActivationOptions())
activate!(y, DataVariableActivationOptions())
foreach(n -> activate!(n, FactorNodeActivationOptions(; callbacks)), (prior, likelihood))

subscription = subscribe!(get_stream_of_marginals(x), (q) -> nothing)
new_observation!(y, 2.0)
unsubscribe!(subscription)
mappings
```

Each mapping names its node, its target and the messages its rule takes.

```@docs
ReactiveMP.MessageMapping
```

A mapping hands the rule the data of the messages and marginals it depends on, keyed as the rule
declares them, with their annotations alongside.

```@docs
ReactiveMP.rule_arguments
```
