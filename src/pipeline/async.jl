export AsyncPipelineStage

import Rocket: async

"""
    AsyncPipelineStage <: AbstractPipelineStage

Applies the `async()` operator from `Rocket.jl` library to the given pipeline
"""
struct AsyncPipelineStage <: AbstractPipelineStage end

apply_pipeline_stage(::AsyncPipelineStage, factornode, tag, stream) = stream |> async()
