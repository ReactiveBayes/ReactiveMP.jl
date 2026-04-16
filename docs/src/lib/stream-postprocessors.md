# [Stream postprocessors](@id lib-stream-postprocessors)

A **stream postprocessor** is a composable transformation applied to one of the reactive observables produced during graph [Activation](@ref lib-node-activation). It wraps a Rocket.jl observable and returns a new observable of the same element type, leaving the message passing logic itself untouched.

The same postprocessor can be applied to three different kinds of streams produced by the inference engine:

- streams of **outbound messages** leaving a factor node interface or a leg of an [`ReactiveMP.EqualityChain`](@ref);
- streams of **marginals** emitted by a [`RandomVariable`](@ref) or by the local cluster of a factor node;
- streams of **scores** (free-energy contributions) used to assemble Bethe Free Energy.

Stream postprocessors are useful for:

- **Scheduling** — controlling *when* downstream subscribers observe updates (e.g. batching a wave of inbound observations into a single propagation step using a `PendingScheduler`, or moving work onto a worker thread using an `AsyncScheduler`).
- **Custom instrumentation** — applying any Rocket.jl operator (filtering, sampling, side-effects) on top of every stream produced by activation.

!!! note
    The previous `AbstractPipelineStage` API and the per-node `scheduler` argument have been unified into [`ReactiveMP.AbstractStreamPostprocessor`](@ref). The old `LoggerPipelineStage` is gone — equivalent behaviour can now be achieved through [callbacks](@ref lib-callbacks) without subscribing to the streams themselves. The migration guide also covers this change.

## [Available stream postprocessors](@id lib-stream-postprocessors-available)

| Postprocessor | Purpose |
|---------------|---------|
| `nothing` | No-op; the implicit default when no postprocessor is attached. The three `postprocess_stream_of_*` methods all have a `::Nothing` fallback that returns the stream unchanged. |
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

Stream postprocessors are provided when activating a factor node via [`ReactiveMP.FactorNodeActivationOptions`](@ref) and a random variable via [`ReactiveMP.RandomVariableActivationOptions`](@ref). In practice this is done through the model specification layer (e.g. [RxInfer.jl](https://github.com/ReactiveBayes/RxInfer.jl)'s `@model` macro), but at the low level it looks like:

```julia
postprocessor = ScheduleOnStreamPostprocessor(PendingScheduler())

# For a factor node
options = ReactiveMP.FactorNodeActivationOptions(
    metadata,
    dependencies,
    postprocessor,   # <-- attached to all streams produced for this node
    annotations,
    rulefallback,
    callbacks,
)
ReactiveMP.activate!(node, options)

# For a random variable
ReactiveMP.activate!(
    var,
    ReactiveMP.RandomVariableActivationOptions(
        postprocessor,
        ReactiveMP.MessageProductContext(),
        ReactiveMP.MessageProductContext(),
    ),
)
```

The same postprocessor instance is applied to every outbound message stream, every marginal stream, and every score stream produced by these activations. A subtype of [`ReactiveMP.AbstractStreamPostprocessor`](@ref) must therefore implement every `postprocess_stream_of_*` method that the kinds of streams it is attached to will go through; to opt out for a particular kind of stream, just forward the stream unchanged.

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

If a postprocessor is attached to a stream whose corresponding `postprocess_stream_of_*` method is not implemented for it, a `MethodError` is raised at activation time. To pass a kind of stream through unchanged, simply return the input stream as shown above.

## API reference

```@docs
ReactiveMP.AbstractStreamPostprocessor
ReactiveMP.postprocess_stream_of_outbound_messages
ReactiveMP.postprocess_stream_of_marginals
ReactiveMP.postprocess_stream_of_scores
ReactiveMP.CompositeStreamPostprocessor
ReactiveMP.ScheduleOnStreamPostprocessor
```
