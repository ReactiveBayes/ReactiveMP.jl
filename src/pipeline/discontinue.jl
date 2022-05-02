export DiscontinuePipelineStage

"""
    DiscontinuePipelineStage <: AbstractPipelineStage

Applies the `discontinue()` operator from `Rocket.jl` library to the given pipeline

See also: [`AbstractPipelineStage`](@ref), [`apply_pipeline_stage`](@ref), [`EmptyPipelineStage`](@ref), [`CompositePipelineStage`](@ref)
"""
struct DiscontinuePipelineStage <: AbstractPipelineStage end

apply_pipeline_stage(::DiscontinuePipelineStage, factornode, tag, stream) = stream |> discontinue()
