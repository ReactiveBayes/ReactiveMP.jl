"""
    AbstractStreamPostprocessor

Abstract supertype for **stream postprocessors** — composable transformations
applied to the reactive observables produced during graph activation.

A stream postprocessor wraps a Rocket.jl observable and returns a new observable
of the same element type. The same postprocessor can be applied to three
different kinds of streams produced by the inference engine, each with its own
entry point:

- [`ReactiveMP.postprocess_stream_of_outbound_messages`](@ref) — the stream of
  outbound [`Message`](@ref)s leaving a factor node interface (or a leg of an
  [`ReactiveMP.EqualityChain`](@ref)).
- [`ReactiveMP.postprocess_stream_of_marginals`](@ref) — the stream of
  [`Marginal`](@ref)s emitted by a [`RandomVariable`](@ref) or by the local
  cluster of a factor node.
- [`ReactiveMP.postprocess_stream_of_scores`](@ref) — the stream of free-energy
  contributions emitted by [`ReactiveMP.score`](@ref).

Stream postprocessors are attached to an inference run via
[`ReactiveMP.FactorNodeActivationOptions`](@ref) and
[`ReactiveMP.RandomVariableActivationOptions`](@ref). Multiple postprocessors
can be chained with [`ReactiveMP.CompositeStreamPostprocessor`](@ref).

# Built-in implementations

- [`ReactiveMP.ScheduleOnStreamPostprocessor`](@ref) — redirects the
  computation onto a custom Rocket.jl scheduler (e.g. `PendingScheduler`,
  `AsyncScheduler`).
- [`ReactiveMP.CompositeStreamPostprocessor`](@ref) — applies a sequence of
  postprocessors in order.

See also: [`ReactiveMP.postprocess_stream_of_outbound_messages`](@ref),
[`ReactiveMP.postprocess_stream_of_marginals`](@ref),
[`ReactiveMP.postprocess_stream_of_scores`](@ref).
"""
abstract type AbstractStreamPostprocessor end

"""
    postprocess_stream_of_outbound_messages(postprocessor, stream)

Apply `postprocessor` to a stream of outbound [`Message`](@ref)s and return the
transformed stream. Called by [`ReactiveMP.activate!`](@ref) on every outbound
message stream produced by a factor node interface.

The default fallback for `::Nothing` returns `stream` unchanged. Subtypes of
[`ReactiveMP.AbstractStreamPostprocessor`](@ref) may override this method to
e.g. redirect emissions to a Rocket.jl scheduler.
"""
function postprocess_stream_of_outbound_messages end

"""
    postprocess_stream_of_marginals(postprocessor, stream)

Apply `postprocessor` to a stream of [`Marginal`](@ref)s and return the
transformed stream. Called by [`ReactiveMP.activate!`](@ref) on every marginal
stream produced for a [`RandomVariable`](@ref) or for a local cluster of a
factor node.

The default fallback for `::Nothing` returns `stream` unchanged. Subtypes of
[`ReactiveMP.AbstractStreamPostprocessor`](@ref) may override this method.
"""
function postprocess_stream_of_marginals end

"""
    postprocess_stream_of_scores(postprocessor, stream)

Apply `postprocessor` to a stream of free-energy score contributions and return
the transformed stream. Called by [`ReactiveMP.score`](@ref) on the per-node
and per-variable score streams used to assemble Bethe Free Energy.

The default fallback for `::Nothing` returns `stream` unchanged. Subtypes of
[`ReactiveMP.AbstractStreamPostprocessor`](@ref) may override this method.
"""
function postprocess_stream_of_scores end

"""
    postprocess_stream_of_outbound_messages(::Nothing, stream) = stream

Pass-through fallback: when no stream postprocessor is configured, outbound
message streams are returned unchanged.
"""
postprocess_stream_of_outbound_messages(::Nothing, stream) = stream

"""
    postprocess_stream_of_marginals(::Nothing, stream) = stream

Pass-through fallback: when no stream postprocessor is configured, marginal
streams are returned unchanged.
"""
postprocess_stream_of_marginals(::Nothing, stream) = stream

"""
    postprocess_stream_of_scores(::Nothing, stream) = stream

Pass-through fallback: when no stream postprocessor is configured, score
streams are returned unchanged.
"""
postprocess_stream_of_scores(::Nothing, stream) = stream

"""
    CompositeStreamPostprocessor{T} <: AbstractStreamPostprocessor

A [`ReactiveMP.AbstractStreamPostprocessor`](@ref) that applies a sequence of
inner postprocessors in order. The output of stage `i` is fed as the input of
stage `i + 1`, for each of the three stream kinds independently.

# Fields
- `stages::T` — a tuple (or any iterable) of postprocessors to apply in order.

# Example

```julia
composite = CompositeStreamPostprocessor((
    ScheduleOnStreamPostprocessor(PendingScheduler()),
    MyCustomPostprocessor(),
))
```

See also: [`ReactiveMP.postprocess_stream_of_outbound_messages`](@ref),
[`ReactiveMP.postprocess_stream_of_marginals`](@ref),
[`ReactiveMP.postprocess_stream_of_scores`](@ref).
"""
struct CompositeStreamPostprocessor{T} <: AbstractStreamPostprocessor
    stages::T
end

function postprocess_stream_of_outbound_messages(
    composite::CompositeStreamPostprocessor, stream
)
    return reduce(
        (stream, stage) ->
            postprocess_stream_of_outbound_messages(stage, stream),
        composite.stages;
        init = stream,
    )
end

function postprocess_stream_of_marginals(
    composite::CompositeStreamPostprocessor, stream
)
    return reduce(
        (stream, stage) -> postprocess_stream_of_marginals(stage, stream),
        composite.stages;
        init = stream,
    )
end

function postprocess_stream_of_scores(
    composite::CompositeStreamPostprocessor, stream
)
    return reduce(
        (stream, stage) -> postprocess_stream_of_scores(stage, stream),
        composite.stages;
        init = stream,
    )
end
