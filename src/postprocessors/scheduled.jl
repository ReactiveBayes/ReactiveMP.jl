import Rocket: release!

"""
    ScheduleOnStreamPostprocessor{S} <: AbstractStreamPostprocessor

A [`ReactiveMP.AbstractStreamPostprocessor`](@ref) that redirects every emission
of the wrapped stream onto a Rocket.jl scheduler via the `schedule_on(scheduler)`
operator. This is the standard way to control *when* downstream subscribers
observe updates — for example, to batch a wave of inbound observations into a
single propagation step using a `PendingScheduler`, or to move work onto a
worker thread using an `AsyncScheduler`.

The same scheduler is applied to all three stream kinds (outbound messages,
marginals, scores), which makes `ScheduleOnStreamPostprocessor` the direct
successor of the v5/early-v6 `ScheduleOnPipelineStage` + node-level scheduler
pair.

# Fields
- `scheduler::S` — a Rocket.jl scheduler. Must be compatible with
  `Rocket.schedule_on`.

# Releasing scheduled updates

If the wrapped scheduler buffers updates (e.g. `PendingScheduler`), call
`Rocket.release!` on the postprocessor to flush them. `release!` is also
defined for tuples and arrays of `ScheduleOnStreamPostprocessor`s for
convenience.

See also: [`ReactiveMP.AbstractStreamPostprocessor`](@ref),
[`ReactiveMP.CompositeStreamPostprocessor`](@ref).
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
