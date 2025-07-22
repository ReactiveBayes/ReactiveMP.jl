export LoggerPipelineStage

"""
    LoggerPipelineStage <: AbstractPipelineStage

Logs all updates from `stream` into `output`

# Arguments 
- `output`: (optional), an output stream used to print log statements
"""
struct LoggerPipelineStage{T} <: AbstractPipelineStage
    output::T
    prefix::String
end

LoggerPipelineStage()               = LoggerPipelineStage(Core.stdout, "Log")
LoggerPipelineStage(output::IO)     = LoggerPipelineStage(output, "Log")
LoggerPipelineStage(prefix::String) = LoggerPipelineStage(Core.stdout, prefix)

logger_pipeline_stage_println(logger::LoggerPipelineStage, something::Any) = logger_pipeline_stage_println(
    logger, logger.output, logger_pipeline_stage_append_prefix(logger, something)
)

logger_pipeline_stage_println(logger::LoggerPipelineStage, output::Core.CoreSTDOUT, something) = Core.println(output, something)
logger_pipeline_stage_println(logger::LoggerPipelineStage, output, something) = println(output, something)

logger_pipeline_stage_append_prefix(logger::LoggerPipelineStage, something) = lazy"[$(logger.prefix)]$something"

apply_pipeline_stage(logger::LoggerPipelineStage, factornode, tag::Val{T}, stream) where {T}             = stream |> tap((v) -> logger_pipeline_stage_println(logger, lazy"[$(functionalform(factornode))][$(T)]: $v"))
apply_pipeline_stage(logger::LoggerPipelineStage, factornode, tag::Tuple{Val{T}, Int}, stream) where {T} = stream |> tap((v) -> logger_pipeline_stage_println(logger, lazy"[$(functionalform(factornode))][$(T):$(tag[2])]: $v"))
