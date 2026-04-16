"""
TODO
"""
abstract type AbstractStreamPostprocessor end

"""
TODO
"""
function postprocess_stream_of_outbound_messages end

"""
TODO
"""
function postprocess_stream_of_marginals end

"""
TODO
"""
function postprocess_stream_of_scores end

"""
TODO
"""
postprocess_stream_of_outbound_messages(::Nothing, stream) = stream

"""
TODO
"""
postprocess_stream_of_marginals(::Nothing, stream) = stream

"""
TODO
"""
postprocess_stream_of_scores(::Nothing, stream) = stream

"""
TODO
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
