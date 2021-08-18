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
    prefix :: String
end

LoggerPipelineStage()               = LoggerPipelineStage(Core.stdout, "Log")
LoggerPipelineStage(output::IO)     = LoggerPipelineStage(output, "Log")
LoggerPipelineStage(prefix::String) = LoggerPipelineStage(Core.stdout, prefix)

apply_pipeline_stage(stage::LoggerPipelineStage, factornode, tag::Type{ <: Val{ T } },    stream) where { T } = stream |> tap((v) -> Core.println(stage.output, "[$(stage.prefix)][$(functionalform(factornode))][$(T)]: $v"))
apply_pipeline_stage(stage::LoggerPipelineStage, factornode, tag::Tuple{ Val{ T }, Int }, stream) where { T } = stream |> tap((v) -> Core.println(stage.output, "[$(stage.prefix)][$(functionalform(factornode))][$(T):$(tag[2])]: $v"))