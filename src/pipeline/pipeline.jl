export AbstractPipelineStage, EmptyPipelineStage, CompositePipelineStage, apply_pipeline_stage

import Base: +

## Abstract Custom Pipeline Stage

"""
    AbstractPipelineStage

An abstract type for all custom pipelines stages

See also: [`apply_pipeline_stage`](@ref), [`EmptyPipelineStage`](@ref), [`CompositePipelineStage`](@ref)
"""
abstract type AbstractPipelineStage end

"""
    apply_pipeline_stage(stage, factornode, tag, stream)

Applies a given pipeline stage to the `stream` argument given `factornode` and `tag` of an edge.

See also: [`AbstractPipelineStage`](@ref), [`EmptyPipelineStage`](@ref), [`CompositePipelineStage`](@ref)
"""
function apply_pipeline_stage end

## Default empty pipeline

"""
    EmptyPipelineStage <: AbstractPipelineStage

Dummy empty pipeline stage that does not modify the original pipeline.

See also: [`AbstractPipelineStage`](@ref), [`apply_pipeline_stage`](@ref), [`CompositePipelineStage`](@ref)
"""
struct EmptyPipelineStage <: AbstractPipelineStage end

apply_pipeline_stage(::EmptyPipelineStage, factornode, tag, stream) = stream

## Composite pipeline

"""
    CompositePipelineStage{T} <: AbstractPipelineStage

Composite pipeline stage that consists of multiple inner pipeline stages 

See also: [`AbstractPipelineStage`](@ref), [`apply_pipeline_stage`](@ref), [`EmptyPipelineStage`](@ref)
"""
struct CompositePipelineStage{T} <: AbstractPipelineStage
    stages::T
end

apply_pipeline_stage(composite::CompositePipelineStage, factornode, tag, stream) =
    reduce((stream, stage) -> apply_pipeline_stage(stage, factornode, tag, stream), composite.stages; init = stream)

Base.:+(stage::AbstractPipelineStage) = stage

Base.:+(left::EmptyPipelineStage, right::EmptyPipelineStage)         = EmptyPipelineStage()
Base.:+(left::EmptyPipelineStage, right::AbstractPipelineStage)      = right
Base.:+(left::AbstractPipelineStage, right::EmptyPipelineStage)      = left
Base.:+(left::AbstractPipelineStage, right::AbstractPipelineStage)   = CompositePipelineStage((left, right))
Base.:+(left::AbstractPipelineStage, right::CompositePipelineStage)  = CompositePipelineStage((left, right.stages...))
Base.:+(left::CompositePipelineStage, right::AbstractPipelineStage)  = CompositePipelineStage((left.stages..., right))
Base.:+(left::CompositePipelineStage, right::CompositePipelineStage) = CompositePipelineStage((left.stages..., right.stages...))

collect_pipeline(any, ::Nothing) = EmptyPipelineStage()
collect_pipeline(any, something) = something