# Rocket.jl message passing specific extensions

using Rocket

mutable struct LimitStackSchedulerProps
    soft_depth :: Int
    hard_depth :: Int
end

struct LimitStackScheduler <: Rocket.AbstractScheduler
    soft_limit :: Int
    hard_limit :: Int
    props      :: LimitStackSchedulerProps
end

LimitStackScheduler(soft_limit::Int)                  = LimitStackScheduler(soft_limit, typemax(Int) - 1)
LimitStackScheduler(soft_limit::Int, hard_limit::Int) = LimitStackScheduler(soft_limit, hard_limit, LimitStackSchedulerProps(0, 0))

get_soft_limit(scheduler::LimitStackScheduler) = scheduler.soft_limit
get_hard_limit(scheduler::LimitStackScheduler) = scheduler.hard_limit

function increase_depth!(scheduler::LimitStackScheduler) 
    scheduler.props.soft_depth = scheduler.props.soft_depth + 1
    scheduler.props.hard_depth = scheduler.props.hard_depth + 1
end

function decrease_depth!(scheduler::LimitStackScheduler) 
    scheduler.props.soft_depth = scheduler.props.soft_depth - 1
    scheduler.props.hard_depth = scheduler.props.hard_depth - 1
end

get_soft_depth(scheduler::LimitStackScheduler)     = scheduler.props.soft_depth
set_soft_depth!(scheduler::LimitStackScheduler, v) = scheduler.props.soft_depth = v

get_hard_depth(scheduler::LimitStackScheduler)     = scheduler.props.hard_depth
set_hard_depth!(scheduler::LimitStackScheduler, v) = scheduler.props.hard_depth = v

Base.show(io::IO, scheduler::LimitStackScheduler) = print(io, "LimitStackScheduler(soft_limit = $(get_soft_limit(scheduler)), hard_limit = $(get_hard_limit(scheduler)))")

Base.similar(scheduler::LimitStackScheduler) = LimitStackScheduler(get_soft_limit(scheduler), get_hard_limit(scheduler))

Rocket.makeinstance(::Type, scheduler::LimitStackScheduler) = scheduler

Rocket.instancetype(::Type, ::Type{ <: LimitStackScheduler }) = LimitStackScheduler

macro limitstack(no_limit_cb, limit_cb)
    output = quote
        increase_depth!(instance)

        if get_hard_depth(instance) >= get_hard_limit(instance)
            error("Hard limit in LimitStackScheduler exceeded")
        end

        result = if get_soft_depth(instance) < get_soft_limit(instance)
            begin 
                $no_limit_cb
            end
        else
            previous_soft_depth = get_soft_depth(instance)
            set_soft_depth!(instance, 0)
            r = begin 
                $limit_cb
            end
            set_soft_depth!(instance, previous_soft_depth)
            r
        end

        decrease_depth!(instance)

        result
    end
    return esc(output)
end

struct LimitStackSubscription <: Teardown
    instance     :: LimitStackScheduler
    subscription :: Teardown
end

Rocket.as_teardown(::Type{ <: LimitStackSubscription }) = UnsubscribableTeardownLogic()

function Rocket.on_unsubscribe!(l::LimitStackSubscription)
    instance     = l.instance
    subscription = l.subscription
    @limitstack Rocket.unsubscribe!(subscription) begin
        @sync @async Rocket.unsubscribe!(subscription)
    end
    return nothing
end

function Rocket.scheduled_subscription!(source, actor, instance::LimitStackScheduler) 
    subscription = @limitstack Rocket.on_subscribe!(source, actor, instance) begin 
        condition  = Base.Condition()
        @async begin
            try
                subscription = Rocket.on_subscribe!(source, actor, instance)
                notify(condition, subscription)
            catch exception
                notify(condition, exception, error = true)
            end
        end
        wait(condition)
    end
    return LimitStackSubscription(instance, subscription)
end

function Rocket.scheduled_next!(actor, value, instance::LimitStackScheduler) 
    @limitstack Rocket.on_next!(actor, value) begin
        @sync @async Rocket.scheduled_next!(actor, value, instance)
    end
end

function Rocket.scheduled_error!(actor, err, instance::LimitStackScheduler) 
    @limitstack Rocket.on_error!(actor, err) begin
        @sync @async Rocket.scheduled_error!(actor, err, instance)
    end
end

function Rocket.scheduled_complete!(actor, instance::LimitStackScheduler) 
    @limitstack Rocket.on_complete!(actor) begin
        @sync @async Rocket.scheduled_complete!(actor, instance)
    end
end