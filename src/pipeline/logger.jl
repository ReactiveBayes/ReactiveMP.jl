export LoggerPipelineStage

"""
    LoggerPipelineStage <: AbstractPipelineStage

Logs all updates from `stream` into `output`

# Arguments 
- `output`: (optional), an output stream used to print log statements

See also: [`AbstractPipelineStage`](@ref), [`apply_pipeline_stage`](@ref), [`EmptyPipelineStage`](@ref), [`CompositePipelineStage`](@ref)
"""
struct LoggerPipelineStage{T} <: AbstractPipelineStage
    output::T
    prefix::String
end

LoggerPipelineStage()               = LoggerPipelineStage(Core.stdout, "Log")
LoggerPipelineStage(output::IO)     = LoggerPipelineStage(output, "Log")
LoggerPipelineStage(prefix::String) = LoggerPipelineStage(Core.stdout, prefix)

Base.println(stage::LoggerPipelineStage, something) = Base.println(stage, stage.output, append_prefix(stage, something))

Base.println(stage::LoggerPipelineStage, output::Core.CoreSTDOUT, something) = Core.println(output, something)
Base.println(stage::LoggerPipelineStage, output, something) = println(output, something)

append_prefix(stage::LoggerPipelineStage, something) = string("[", stage.prefix, "]", something)

apply_pipeline_stage(stage::LoggerPipelineStage, factornode, tag::Val{T}, stream) where {T}             = stream |> tap((v) -> println(stage, "[$(functionalform(factornode))][$(T)]: $v"))
apply_pipeline_stage(stage::LoggerPipelineStage, factornode, tag::Tuple{Val{T}, Int}, stream) where {T} = stream |> tap((v) -> println(stage, "[$(functionalform(factornode))][$(T):$(tag[2])]: $v"))
