export ScheduleOnPortal, apply, update!, schedule_pending, schedule_updates

import Rocket: release!

struct ScheduleOnPortal{S} <: AbstractPortal
    scheduler :: S
end

apply(portal::ScheduleOnPortal, factornode, tag, stream) = stream |> schedule_on(portal.scheduler)

Rocket.release!(portal::ScheduleOnPortal)                        = release!(portal.scheduler)
Rocket.release!(portals::NTuple{N, <: ScheduleOnPortal}) where N = foreach(release!, portals)
Rocket.release!(portals::AbstractArray{ <: ScheduleOnPortal })   = foreach(release!, portals)

update!(portal::ScheduleOnPortal)                        = release!(portal.scheduler)
update!(portals::NTuple{N, <: ScheduleOnPortal}) where N = foreach(update!, portals)
update!(portals::AbstractArray{ <: ScheduleOnPortal })   = foreach(update!, portals)

schedule_pending(var::AbstractVariable) = schedule_pending([ var ])

function schedule_pending(vars::NTuple{N, <: AbstractVariable }) where N
    portal = ScheduleOnPortal(PendingScheduler())
    foreach((v) -> inbound_portal!(v, portal), vars)
    return portal
end

function schedule_pending(vars::AbstractArray{ <: AbstractVariable })
    portal = ScheduleOnPortal(PendingScheduler())
    foreach((v) -> inbound_portal!(v, portal), vars)
    return portal
end

schedule_updates(args...) = map(schedule_pending, args)