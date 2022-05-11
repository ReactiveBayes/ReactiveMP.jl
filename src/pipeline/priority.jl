export PriorityPipelineStage

"""
    PriorityPipelineStage <: AbstractPipelineStage

Applies the `schedule_on()` operator from `Rocket.jl` library with the corresponding `PriorityScheduler`

See also: [`AbstractPipelineStage`](@ref), [`apply_pipeline_stage`](@ref), [`EmptyPipelineStage`](@ref), [`CompositePipelineStage`](@ref)
"""
struct PriorityPipelineStage{S <: PriorityScheduler} <: AbstractPipelineStage 
    scheduler :: S
end

function apply_pipeline_stage(stage::PriorityPipelineStage, factornode, ::Type{Val{S}}, stream) where S
    interface = getinterface(factornode, S) 
    variable  = connectedvar(interface)
    varname   = name(variable)
    T         = Rocket.subscribable_extract_type(stream)
    # return stream |> map(Tuple{Symbol, T}, (d) -> (varname, d)) |> schedule_on(stage.scheduler) |> map(T, t -> t[2])
    return stream |> map(Tuple{Symbol, T}, (d) -> (varname, d)) |> schedule_on(stage.scheduler) |> map(T, t -> t[2])
end