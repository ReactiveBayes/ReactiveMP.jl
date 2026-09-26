# [Stream postprocessors](@id lib-stream-postprocessors)

A **stream postprocessor** is a composable transformation of the reactive streams the engine
builds. It wraps a Rocket.jl observable and returns a new one of the same element type, leaving
what the stream computes untouched.

The same postprocessor can apply to three kinds of stream:

- **outbound messages**, of a factor node's interfaces and of a random variable's equality chain;
- **marginals**, of a random variable and of a factor node's joint clusters;
- **scores**, the free-energy contributions [`score`](@ref) builds when it is given a
  postprocessor. Activation builds no score stream, and [`bethe_free_energy`](@ref) applies none.

Stream postprocessors are useful for:

- **Scheduling** — controlling *when* downstream subscribers observe updates (e.g. batching a wave of inbound observations into a single propagation step using a `PendingScheduler`, or moving work onto a worker thread using an `AsyncScheduler`).
- **Custom instrumentation** — applying any Rocket.jl operator (filtering, sampling, side-effects) on top of every stream produced by activation.

!!! note
    To observe what the engine computes without changing the streams, use [callbacks](@ref lib-callbacks) instead.

## [Available stream postprocessors](@id lib-stream-postprocessors-available)

| Postprocessor | Purpose |
|---------------|---------|
| `nothing` | None, the default: each `postprocess_stream_of_*` returns the stream unchanged. |
| [`ReactiveMP.ScheduleOnStreamPostprocessor`](@ref) | Redirects every emission to a Rocket.jl scheduler via the `schedule_on(scheduler)` operator. |
| [`ReactiveMP.CompositeStreamPostprocessor`](@ref) | Applies a sequence of postprocessors in order. |

## [Composing stream postprocessors](@id lib-stream-postprocessors-compose)

Multiple postprocessors are chained by wrapping them in a [`ReactiveMP.CompositeStreamPostprocessor`](@ref):

```julia
postprocessor = CompositeStreamPostprocessor((
    ScheduleOnStreamPostprocessor(PendingScheduler()),
    MyCustomStreamPostprocessor(),
))
```

The output of stage `i` is fed as the input of stage `i + 1`, independently for each of the three stream kinds.

## [Attaching a stream postprocessor](@id lib-stream-postprocessors-attach)

A postprocessor is given when a factor node is activated, as the option `postprocessor` of
[`ReactiveMP.FactorNodeActivationOptions`](@ref), and when a random variable is, as the first
field of [`RandomVariableActivationOptions`](@ref). RxInfer does this for a model; with the
engine alone:

```julia
postprocessor = ReactiveMP.ScheduleOnStreamPostprocessor(PendingScheduler())

# every outbound message stream and joint marginal stream of the node
ReactiveMP.activate!(node, ReactiveMP.FactorNodeActivationOptions(; postprocessor))

# the variable's equality chain and its marginal stream
ReactiveMP.activate!(x, RandomVariableActivationOptions(postprocessor, ReactiveMP.MessageProductContext(), ReactiveMP.MessageProductContext()))

# the updates held by the scheduler are delivered when it is released
Rocket.release!(postprocessor)
```

The same instance applies to every stream of these activations. A subtype of
[`ReactiveMP.AbstractStreamPostprocessor`](@ref) therefore implements every
`postprocess_stream_of_*` method of the kinds of stream it is attached to; to leave a kind
unchanged, it returns the stream as it is.

## [Custom stream postprocessors](@id lib-stream-postprocessors-custom)

Custom postprocessors are created by subtyping [`ReactiveMP.AbstractStreamPostprocessor`](@ref) and implementing one or more of [`ReactiveMP.postprocess_stream_of_outbound_messages`](@ref), [`ReactiveMP.postprocess_stream_of_marginals`](@ref), and [`ReactiveMP.postprocess_stream_of_scores`](@ref):

```julia
using Rocket

struct MyStreamPostprocessor <: ReactiveMP.AbstractStreamPostprocessor end

# Postprocess outbound messages — `tap` performs a side effect and forwards
# the value unchanged.
function ReactiveMP.postprocess_stream_of_outbound_messages(::MyStreamPostprocessor, stream)
    return stream |> tap(msg -> println("Intercepted: ", msg))
end

# Pass marginals and scores through unchanged.
ReactiveMP.postprocess_stream_of_marginals(::MyStreamPostprocessor, stream) = stream
ReactiveMP.postprocess_stream_of_scores(::MyStreamPostprocessor, stream)    = stream
```

A postprocessor attached to a kind of stream it has no method for is a `MethodError` at activation. To pass a kind of stream through unchanged, return it, as above.

## API reference

```@docs
ReactiveMP.AbstractStreamPostprocessor
ReactiveMP.postprocess_stream_of_outbound_messages
ReactiveMP.postprocess_stream_of_marginals
ReactiveMP.postprocess_stream_of_scores
ReactiveMP.CompositeStreamPostprocessor
ReactiveMP.ScheduleOnStreamPostprocessor
```
