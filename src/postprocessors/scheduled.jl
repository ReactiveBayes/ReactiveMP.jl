import Rocket: release!

"""
TODO
"""
struct ScheduleOnStreamPostprocessor{S} <: AbstractStreamPostprocessor
    scheduler::S
end

postprocess_stream_of_outbound_messages(
    p::ScheduleOnStreamPostprocessor, stream
) = stream |> schedule_on(p.scheduler)

postprocess_stream_of_marginals(p::ScheduleOnStreamPostprocessor, stream) =
    stream |> schedule_on(p.scheduler)

postprocess_stream_of_scores(p::ScheduleOnStreamPostprocessor, stream) =
    stream |> schedule_on(p.scheduler)

Rocket.release!(stage::ScheduleOnStreamPostprocessor)                         = release!(stage.scheduler)
Rocket.release!(stages::NTuple{N, <:ScheduleOnStreamPostprocessor}) where {N} = foreach(release!, stages)
Rocket.release!(stages::AbstractArray{<:ScheduleOnStreamPostprocessor})       = foreach(release!, stages)
