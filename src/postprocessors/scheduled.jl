import Rocket: release!

"""
    ReactiveMP.ScheduleOnStreamPostprocessor(scheduler)

A stream postprocessor that delivers every emission on a Rocket.jl scheduler, with
`schedule_on(scheduler)`, for every kind of stream: it controls when subscribers see updates. A
`PendingScheduler` holds them until they are released, so a wave of observations propagates as
one step; an `AsyncScheduler` moves the work to another task.

# Fields

- `scheduler`: a Rocket.jl scheduler that `Rocket.schedule_on` accepts.

`Rocket.release!(postprocessor)` releases the updates a buffering scheduler holds; it also takes
a tuple or an array of `ScheduleOnStreamPostprocessor`s.
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

Rocket.release!(stage::ScheduleOnStreamPostprocessor) = release!(stage.scheduler)
Rocket.release!(stages::NTuple{N, <:ScheduleOnStreamPostprocessor}) where {N} = foreach(release!, stages)
Rocket.release!(stages::AbstractArray{<:ScheduleOnStreamPostprocessor}) = foreach(release!, stages)
