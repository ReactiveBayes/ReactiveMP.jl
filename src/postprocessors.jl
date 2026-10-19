"""
    ReactiveMP.AbstractStreamPostprocessor

The supertype of stream postprocessors: transformations of the streams the engine builds, each
a Rocket.jl observable returned with the same element type, applied without changing what the
streams compute. A postprocessor implements one method per kind of stream it applies to:

- [`ReactiveMP.postprocess_stream_of_outbound_messages`](@ref), for a factor node's outbound
  messages and the streams of a random variable's [`ReactiveMP.EqualityChain`](@ref);
- [`ReactiveMP.postprocess_stream_of_marginals`](@ref), for a random variable's marginal and a
  factor node's joint marginals;
- [`ReactiveMP.postprocess_stream_of_scores`](@ref), for the free-energy contributions a caller of
  [`score`](@ref) asks for.

A postprocessor is given at activation, as the `postprocessor` of
[`ReactiveMP.FactorNodeActivationOptions`](@ref) and the `stream_postprocessor` of
[`RandomVariableActivationOptions`](@ref), and to [`score`](@ref) directly;
[`bethe_free_energy`](@ref) applies none. `nothing` is no postprocessor.

The postprocessors defined here are [`ReactiveMP.ScheduleOnStreamPostprocessor`](@ref) and
[`ReactiveMP.CompositeStreamPostprocessor`](@ref).
"""
abstract type AbstractStreamPostprocessor end

"""
    ReactiveMP.postprocess_stream_of_outbound_messages(postprocessor, stream)

Apply `postprocessor` to a stream of outbound messages and return the new stream. Activation
calls it on every outbound message stream of a factor node, and on the streams of a random
variable's [`ReactiveMP.EqualityChain`](@ref). For `nothing` it returns `stream` as it is.
"""
function postprocess_stream_of_outbound_messages end

"""
    ReactiveMP.postprocess_stream_of_marginals(postprocessor, stream)

Apply `postprocessor` to a stream of [`Marginal`](@ref)s and return the new stream. Activation
calls it on the marginal stream of a random variable and on every joint marginal stream of a
factor node. For `nothing` it returns `stream` as it is.
"""
function postprocess_stream_of_marginals end

"""
    ReactiveMP.postprocess_stream_of_scores(postprocessor, stream)

Apply `postprocessor` to a stream of free-energy contributions and return the new stream.
[`score`](@ref) calls it on the stream it builds, with the postprocessor it is given. For
`nothing` it returns `stream` as it is.
"""
function postprocess_stream_of_scores end

postprocess_stream_of_outbound_messages(::Nothing, stream) = stream

postprocess_stream_of_marginals(::Nothing, stream) = stream

postprocess_stream_of_scores(::Nothing, stream) = stream

"""
    ReactiveMP.CompositeStreamPostprocessor(stages)

A stream postprocessor that applies several in order, the output of each stage the input of the
next, for each kind of stream. Every stage implements the kinds of stream the composite is
applied to.

# Fields

- `stages`: the postprocessors, a tuple or any other iterable.

# Examples

```julia
postprocessor = ReactiveMP.CompositeStreamPostprocessor((
    ReactiveMP.ScheduleOnStreamPostprocessor(PendingScheduler()),
    MyStreamPostprocessor(),
))
```
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
