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
struct NoopStreamPostprocessor <: AbstractStreamPostprocessor end

"""
TODO
"""
postprocess_stream_of_outbound_messages(::NoopStreamPostprocessor, stream) =
    stream

"""
TODO
"""
postprocess_stream_of_marginals(::NoopStreamPostprocessor, stream) = stream

"""
TODO
"""
postprocess_stream_of_scores(::NoopStreamPostprocessor, stream) = stream

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

import Base: +

Base.:+(stage::AbstractStreamPostprocessor) = stage

Base.:+(left::NoopStreamPostprocessor, right::NoopStreamPostprocessor)           = NoopStreamPostprocessor()
Base.:+(left::NoopStreamPostprocessor, right::AbstractStreamPostprocessor)       = right
Base.:+(left::AbstractStreamPostprocessor, right::NoopStreamPostprocessor)       = left
Base.:+(left::AbstractStreamPostprocessor, right::AbstractStreamPostprocessor)   = CompositeStreamPostprocessor((left, right))
Base.:+(left::AbstractStreamPostprocessor, right::CompositeStreamPostprocessor)  = CompositeStreamPostprocessor((left, right.stages...))
Base.:+(left::CompositeStreamPostprocessor, right::AbstractStreamPostprocessor)  = CompositeStreamPostprocessor((left.stages..., right))
Base.:+(left::CompositeStreamPostprocessor, right::CompositeStreamPostprocessor) = CompositeStreamPostprocessor((left.stages..., right.stages...))

"""
    as_stream_postprocessor(nodetype, stream_postprocessor)

This function converts given postprocessor to a correct internal postprocessor representation for a given factor node.
Typically simply returns the provided `stream_postprocessor`. If `stream_postprocessor` is `nothing`, 
returns [`ReactiveMP.NoopStreamPostprocessor`](@ref).
"""
function as_stream_postprocessor end

as_stream_postprocessor(_, ::Nothing) = NoopStreamPostprocessor()
as_stream_postprocessor(_, something) = something
