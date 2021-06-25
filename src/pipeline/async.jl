export AsyncPipelineStage

import Rocket: async

"""
    AsyncPipelineStage <: AbstractPipelineStage

Applies the `async()` operator from `Rocket.jl` library to the given pipeline

See also: [`AbstractPipelineStage`](@ref), [`apply_pipeline_stage`](@ref), [`EmptyPipelineStage`](@ref), [`CompositePipelineStage`](@ref)
"""
struct AsyncPipelineStage <: AbstractPipelineStage end

apply_pipeline_stage(::AsyncPipelineStage, factornode, tag, stream) = stream |> async()