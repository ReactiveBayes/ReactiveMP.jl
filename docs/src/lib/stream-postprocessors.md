# [Stream postprocessors](@id lib-stream-postprocessors)

A stream postprocessor transforms the reactive streams the engine builds. It takes a Rocket.jl
observable and returns a new one of the same element type, and it leaves what the stream computes
unchanged. It applies to three kinds of stream:

- **outbound messages**, of a factor node's interfaces and of a random variable's equality chain;
- **marginals**, of a random variable and of a factor node's joint clusters;
- **scores**, the free-energy terms that [`score`](@ref) builds when you give it a postprocessor.
  Activation builds no score stream, and [`bethe_free_energy`](@ref) applies no postprocessor.

A postprocessor controls *when* subscribers see updates, or adds any Rocket.jl operator on top of
every stream that activation builds. To observe what the engine computes without changing the
streams, use [callbacks](@ref lib-callbacks) instead.

```@setup postprocessors
using ReactiveMP, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, Rocket
import ReactiveMP: activate!, FactorNodeActivationOptions, MessageProductContext, get_stream_of_marginals
include(joinpath(pkgdir(ReactiveMP), "docs", "nodes.jl"))
```

The examples use the `Gaussian` node of [The example node](@ref example-node). Each builds a
latent `x` with a normal prior, observed through a second node, and activates both nodes with a
postprocessor:

```@example postprocessors
function graph(postprocessor)
    x, y = randomvar(label = :x), datavar(label = :y)
    prior = factornode(Gaussian, [(:out, x), (:μ, constvar(0.0)), (:v, constvar(10.0))])
    likelihood = factornode(Gaussian, [(:out, y), (:μ, x), (:v, constvar(1.0))])
    activate!(x, RandomVariableActivationOptions(postprocessor, MessageProductContext(), MessageProductContext()))
    activate!(y, DataVariableActivationOptions())
    foreach(n -> activate!(n, FactorNodeActivationOptions(; postprocessor)), (prior, likelihood))
    return x, y
end
nothing # hide
```

## [Available stream postprocessors](@id lib-stream-postprocessors-available)

| Postprocessor | Purpose |
|---------------|---------|
| `nothing` | None, the default: each `postprocess_stream_of_*` returns the stream unchanged. |
| [`ReactiveMP.ScheduleOnStreamPostprocessor`](@ref) | Delivers every emission through a Rocket.jl scheduler, with the `schedule_on(scheduler)` operator. |
| [`ReactiveMP.CompositeStreamPostprocessor`](@ref) | Applies several postprocessors in order. |

A `PendingScheduler` holds every update until you release it. This batches a wave of
observations into a single propagation step:

```@example postprocessors
postprocessor = ReactiveMP.ScheduleOnStreamPostprocessor(PendingScheduler())
x, y = graph(postprocessor)
subscription = subscribe!(get_stream_of_marginals(x), (q) -> println("q(x) = ", q))

new_observation!(y, 2.0)
println("observed")
Rocket.release!(postprocessor)
println("released once")
Rocket.release!(postprocessor)
unsubscribe!(subscription)
```

The scheduler holds the nodes' messages until the first release. Their product, the marginal,
passes through the scheduler too, and arrives at the second release. An `AsyncScheduler` moves
the work onto another task instead.

## [Attaching a stream postprocessor](@id lib-stream-postprocessors-attach)

You give a postprocessor when you activate a factor node, as the option `postprocessor` of
[`ReactiveMP.FactorNodeActivationOptions`](@ref), and when you activate a random variable, as the
first field of [`RandomVariableActivationOptions`](@ref). The `graph` function above does both.
On a node, it applies to every outbound message stream and joint marginal stream. On a variable,
it applies to the equality chain and the marginal stream. RxInfer attaches a model's
postprocessor the same way.

The same instance applies to every stream of these activations. A subtype of
[`ReactiveMP.AbstractStreamPostprocessor`](@ref) therefore implements the
`postprocess_stream_of_*` method of every kind of stream it is attached to.

## [Custom stream postprocessors](@id lib-stream-postprocessors-custom)

A custom postprocessor subtypes [`ReactiveMP.AbstractStreamPostprocessor`](@ref) and implements
[`ReactiveMP.postprocess_stream_of_outbound_messages`](@ref),
[`ReactiveMP.postprocess_stream_of_marginals`](@ref) and
[`ReactiveMP.postprocess_stream_of_scores`](@ref). To leave a kind of stream unchanged, a method
returns it as it is. The postprocessor below prints every outbound message with Rocket's `tap`,
which runs a side effect and forwards the value:

```@example postprocessors
struct PrintMessages <: ReactiveMP.AbstractStreamPostprocessor end

ReactiveMP.postprocess_stream_of_outbound_messages(::PrintMessages, stream) =
    stream |> tap((message) -> println("message: ", getdata(as_message(message))))
ReactiveMP.postprocess_stream_of_marginals(::PrintMessages, stream) = stream
ReactiveMP.postprocess_stream_of_scores(::PrintMessages, stream) = stream

x, y = graph(PrintMessages())
subscription = subscribe!(get_stream_of_marginals(x), (q) -> println("q(x) = ", q))
new_observation!(y, 2.0)
unsubscribe!(subscription)
```

A node emits a [`DeferredMessage`](@ref), which [`as_message`](@ref) computes. The two messages
are the prior's and the likelihood's, towards `x`. A postprocessor attached to a kind of stream
it has no method for is a `MethodError` at activation.

## [Composing stream postprocessors](@id lib-stream-postprocessors-compose)

A [`ReactiveMP.CompositeStreamPostprocessor`](@ref) chains several postprocessors. The output of
stage `i` is the input of stage `i + 1`, for each of the three kinds of stream:

```@example postprocessors
pending = ReactiveMP.ScheduleOnStreamPostprocessor(PendingScheduler())
x, y = graph(ReactiveMP.CompositeStreamPostprocessor((PrintMessages(), pending)))
subscription = subscribe!(get_stream_of_marginals(x), (q) -> println("q(x) = ", q))

new_observation!(y, 2.0)
println("observed")
Rocket.release!(pending)
Rocket.release!(pending)
unsubscribe!(subscription)
```

The messages print as the nodes emit them, before the scheduler holds them, and the posterior
waits for the two releases.

## API reference

```@docs
ReactiveMP.AbstractStreamPostprocessor
ReactiveMP.postprocess_stream_of_outbound_messages
ReactiveMP.postprocess_stream_of_marginals
ReactiveMP.postprocess_stream_of_scores
ReactiveMP.CompositeStreamPostprocessor
ReactiveMP.ScheduleOnStreamPostprocessor
```
