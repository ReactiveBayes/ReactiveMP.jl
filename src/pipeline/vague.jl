export InitVaguePipelineStage

# TODO, wont' work for nodes with different variate_form on different edges

"""
    InitVaguePipelineStage{D} <: AbstractPipelineStage 

Uses the `start_with()` operator from `Rocket.jl` library and `vague` function to initialise first updates for a `stream` in the form of vague messages.

# Arguments 
- `dimensionality`: (optional) dimensionality of initial vague messages

See also: [`AbstractPipelineStage`](@ref), [`apply_pipeline_stage`](@ref), [`EmptyPipelineStage`](@ref), [`CompositePipelineStage`](@ref)
"""
struct InitVaguePipelineStage{D} <: AbstractPipelineStage 
    dimensionality :: D
end

InitVaguePipelineStage() = InitVaguePipelineStage(nothing)

getdimensionality(stage::InitVaguePipelineStage) = stage.dimensionality

variate_form(stage::InitVaguePipelineStage) = variate_form(stage, getdimensionality(stage))

variate_form(::InitVaguePipelineStage, ::Nothing)         = Univariate
variate_form(::InitVaguePipelineStage, ::Int)             = Multivariate
variate_form(::InitVaguePipelineStage, ::Tuple{Int, Int}) = Matrixvariate

apply_pipeline_stage(portal::InitVaguePortal, factornode, tag, stream) = __apply_vague_pipeline_stage(variate_form(portal), portal, factornode, tag, stream)

function __apply_vague_pipeline_stage(::Type{ Univariate }, portal::InitVaguePortal, factornode, tag, stream) 
    return stream |> start_with(Message(vague(conjugate_type(functionalform(factornode), tag)), false, true))
end

function __apply_vague_pipeline_stage(::Type{ Multivariate }, portal::InitVaguePortal, factornode, tag, stream) 
    return stream |> start_with(Message(vague(conjugate_type(functionalform(factornode), tag), getdimensionality(portal)), false, true))
end

function __apply_vague_pipeline_stage(::Type{ Matrixvariate }, portal::InitVaguePortal, factornode, tag, stream) 
    return stream |> start_with(Message(vague(conjugate_type(functionalform(factornode), tag), getdimensionality(portal)...), false, true))
end