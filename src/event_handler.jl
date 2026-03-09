
"""
A simple structure to name an event and enable dispatch on different event types.

```jldoctest
julia> struct MyEventHandler end

julia> ReactiveMP.handle_event(handler::MyEventHandler, event::ReactiveMP.Event{:before_message_rule_call}, args...) = print(args...);

julia> ReactiveMP.broadcast_event(MyEventHandler(), ReactiveMP.Event{:before_message_rule_call}(), 1)
1
```

See also: [`ReactiveMP.handle_event`](@ref)
"""
struct Event{E} end

"""
    handle_event(event_handler, event, args...)

Custom event handlers should implement `handle_event` instead of `broadcast_event`.
A typical event has type [`ReactiveMP.Event`](@ref), e.g. `ReactiveMP.Event{:before_message_rule_call}`.
"""
function handle_event end

"""
    broadcast_event(event_handler, event, args...)

An internal function to broadcast events to the outside world via a registered
`event_handler`. Useful for injecting custom logic into the message passing procedure.

The `event_handler` should implement the [`ReactiveMP.handle_event`] method for different event types.
A typical event has type [`ReactiveMP.Event`](@ref), e.g. `ReactiveMP.Event{:before_message_rule_call}`.

The `event_handler` can also be a `NamedTuple` with fields corresponding to event names, e.g.

```jldoctest
julia> event_handler = (before_message_rule_call = (args...) -> sum(args),);

julia> ReactiveMP.broadcast_event(event_handler, ReactiveMP.Event{:before_message_rule_call}(), 1, 2)
3

julia> ReactiveMP.broadcast_event(event_handler, ReactiveMP.Event{:other_event}(), 1, 2, 3)
```

If the `NamedTuple` does not have a field corresponding to the event name, the event will be ignored.
"""
function broadcast_event end

function broadcast_event(::Nothing, event, args...)
    return nothing
end

function broadcast_event(event_handler, event, args...)
    return handle_event(event_handler, event, args...)
end

"""
    handle_event(event_handler::NamedTuple, event::Event, args...)

A specialized handler for `NamedTuple`s.

See also: [`ReactiveMP.broadcast_event`](@ref), [`ReactiveMP.Event`](@ref)
"""
function handle_event(event_handler::NamedTuple{K}, ::Event{E}, args...) where {K, E}
    if E in K
        return event_handler[E](args...)
    end
    return nothing
end



# All defined events go here, so its easier to document them all in one place

"""
    BeforeMessageRuleCallEvent # Event{:before_message_rule_call}

Alias for `Event{:before_message_rule_call}`. This event that is being fired right 
before computing the message and calling the corresponding rule. The handler for this event 
should accept three arguments in the following order:
- `mapping` of type [`ReactiveMP.MessageMapping`](@ref)
- `messages`, typically of type `Tuple` if present, `nothing` otherwise
- `marginals`, typically of type `Tuple` if present, `nothing` otherwise

```jldoctest
julia> import ReactiveMP: BeforeMessageRuleCallEvent

julia> struct MyEventHandler end

julia> ReactiveMP.handle_event(::MyEventHandler, ::BeforeMessageRuleCallEvent, mapping, messages, marginals) = println("Before message called!")
```
"""
const BeforeMessageRuleCallEvent = Event{:before_message_rule_call}

"""
    AfterMessageRuleCallEvent # Event{:after_message_rule_call}

Alias for `Event{:after_message_rule_call}`. This event that is being fired right 
after computing the message and calling the corresponding rule. The handler for this event 
should accept three arguments in the following order:
- `mapping` of type [`ReactiveMP.MessageMapping`](@ref)
- `messages`, typically of type `Tuple` if present, `nothing` otherwise
- `marginals`, typically of type `Tuple` if present, `nothing` otherwise
- `result`, the result of the rule invocation (of `rulefallback`), can be any type
- `addons`, the result of the addons invocation, if present, can be any type

```jldoctest
julia> import ReactiveMP: AfterMessageRuleCallEvent

julia> struct MyEventHandler end

julia> ReactiveMP.handle_event(::MyEventHandler, ::AfterMessageRuleCallEvent, mapping, messages, marginals) = println("After message called!")
```
"""
const AfterMessageRuleCallEvent = Event{:after_message_rule_call}
