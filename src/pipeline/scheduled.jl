export ScheduleOnPipelineStage, schedule_updates

import Rocket: release!

"""
    ScheduleOnPipelineStage{S} <: AbstractPipelineStage

Applies the `schedule_on()` operator from `Rocket.jl` library to the given pipeline with a provided `scheduler`

# Arguments 
- `scheduler`: scheduler to schedule updates on. Must be compatible with `Rocket.jl` library and `schedule_on()` operator.
"""
struct ScheduleOnPipelineStage{S} <: AbstractPipelineStage
    scheduler::S
end

apply_pipeline_stage(stage::ScheduleOnPipelineStage, factornode, tag, stream) = stream |> schedule_on(stage.scheduler)

Rocket.release!(stage::ScheduleOnPipelineStage)                         = release!(stage.scheduler)
Rocket.release!(stages::NTuple{N, <:ScheduleOnPipelineStage}) where {N} = foreach(release!, stages)
Rocket.release!(stages::AbstractArray{<:ScheduleOnPipelineStage})       = foreach(release!, stages)

update!(stage::ScheduleOnPipelineStage)                         = release!(stage.scheduler)
update!(stages::NTuple{N, <:ScheduleOnPipelineStage}) where {N} = foreach(update!, stages)
update!(stages::AbstractArray{<:ScheduleOnPipelineStage})       = foreach(update!, stages)

function _schedule_updates end

__schedule_updates(var::AbstractVariable)                         = __schedule_updates((var,))
__schedule_updates(vars::NTuple{N, <:AbstractVariable}) where {N} = __schedule_updates(ScheduleOnPipelineStage(PendingScheduler()), vars)
__schedule_updates(vars::AbstractArray{<:AbstractVariable})       = __schedule_updates(ScheduleOnPipelineStage(PendingScheduler()), vars)

__schedule_updates(pipeline_stage::ScheduleOnPipelineStage, var::AbstractVariable) = __schedule_updates(pipeline_stage, (var,))

function __schedule_updates(pipeline_stage::ScheduleOnPipelineStage, vars::NTuple{N, <:AbstractVariable}) where {N}
    foreach((v) -> add_pipeline_stage!(v, pipeline_stage), vars)
    return pipeline_stage
end

function __schedule_updates(pipeline_stage::ScheduleOnPipelineStage, vars::AbstractArray{<:AbstractVariable})
    foreach((v) -> add_pipeline_stage!(v, pipeline_stage), vars)
    return pipeline_stage
end

"""
    schedule_updates(variables...; pipeline_stage = ScheduleOnPipelineStage(PendingScheduler())) 

Schedules posterior marginal updates for given variables using `stage`. By default creates `ScheduleOnPipelineStage` with `PendingScheduler()` from `Rocket.jl` library.
Returns a scheduler with `release!` method available to release all scheduled updates.
"""
function schedule_updates(args...; pipeline_stage = ScheduleOnPipelineStage(PendingScheduler()))
    return map((arg) -> __schedule_updates(pipeline_stage, arg), args)
end
