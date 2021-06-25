export LoggerPipelineStage

"""
    LoggerPipelineStage <: AbstractPipelineStage

Logs all updates from `stream` into `output`

# Arguments 
- `output`: (optional), an output stream used to print log statements

See also: [`AbstractPipelineStage`](@ref), [`apply_pipeline_stage`](@ref), [`EmptyPipelineStage`](@ref), [`CompositePipelineStage`](@ref)
"""
struct LoggerPipelineStage{T} <: AbstractPipelineStage 
    output :: T
end

LoggerPipelineStage() = LoggerPipelineStage(stdout)

apply_pipeline_stage(::LoggerPipelineStage, factornode, tag::Type{ <: Val{ T } },    stream) where { T } = stream |> tap((v) -> Core.println("[Log][$(functionalform(factornode))][$(T)]: $v"))
apply_pipeline_stage(::LoggerPipelineStage, factornode, tag::Tuple{ Val{ T }, Int }, stream) where { T } = stream |> tap((v) -> Core.println("[Log][$(functionalform(factornode))][$(T):$(tag[2])]: $v"))