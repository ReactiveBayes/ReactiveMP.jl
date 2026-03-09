
"""
    invoke_callback(callbacks, event, args...)

Custom callbacks handlers should implement `invoke_callback` in order to listen to events 
during the reactive message passing procedure.
A typical event has type `Val{...}`, e.g. `Val{:before_message_rule_call}`.
Does nothing if `callbacks` is `nothing`.

```jldoctest
julia> struct MyCustomCallbackHandler end;

julia> ReactiveMP.invoke_callback(::MyCustomCallbackHandler, event, args...) = print("Event \$(event) has been called with \$(args)");
```
"""
function invoke_callback(callbacks::Nothing, args...)
    return nothing
end

"""
    invoke_callback(callbacks::NamedTuple, event::Val{...}, args...)

The `callbacks` can also be a `NamedTuple` with fields corresponding to event names, e.g.

```jldoctest
julia> callbacks = (before_message_rule_call = (args...) -> sum(args),);

julia> ReactiveMP.invoke_callback(callbacks , Val{:before_message_rule_call}(), 1, 2)
3

julia> ReactiveMP.invoke_callback(callbacks , Val{:other_event}(), 1, 2, 3)
```

If the `NamedTuple` does not have a field corresponding to the event name, the event will be ignored.
"""
function invoke_callback(callbacks::NamedTuple{K}, ::Val{E}, args...) where {K, E}
    if E in K
        return callbacks[E](args...)
    end
    return nothing
end

# All defined events go here, so its easier to document them all in one place

"""
    BeforeMessageRuleCallback # Val{:before_message_rule_call}

Alias for `Val{:before_message_rule_call}`. This event is being used to call a callback right
before computing the message and calling the corresponding rule. The callback handler for this event
should accept three arguments in the following order:
- `mapping` of type [`ReactiveMP.MessageMapping`](@ref)
- `messages`, typically of type `Tuple` if present, `nothing` otherwise
- `marginals`, typically of type `Tuple` if present, `nothing` otherwise

```jldoctest
julia> import ReactiveMP: BeforeMessageRuleCallback

julia> struct MyCallbackHandler end

julia> ReactiveMP.invoke_callback(::MyCallbackHandler, ::BeforeMessageRuleCallback, mapping, messages, marginals) = println("Before message called!")
```

See also: [`ReactiveMP.invoke_callback`](@ref)
"""
const BeforeMessageRuleCallback = Val{:before_message_rule_call}

"""
    AfterMessageRuleCallback # Val{:after_message_rule_call}

Alias for `Val{:after_message_rule_call}`. This event is being used to call a callback right
after computing the message and calling the corresponding rule. The callback handler for this event 
should accept three arguments in the following order:
- `mapping` of type [`ReactiveMP.MessageMapping`](@ref)
- `messages`, typically of type `Tuple` if present, `nothing` otherwise
- `marginals`, typically of type `Tuple` if present, `nothing` otherwise
- `result`, the result of the rule invocation (of `rulefallback`), can be any type
- `addons`, the result of the addons invocation, if present, can be any type

```jldoctest
julia> import ReactiveMP: AfterMessageRuleCallback

julia> struct MyCallbackHandler end

julia> ReactiveMP.invoke_callback(::MyCallbackHandler, ::AfterMessageRuleCallback, mapping, messages, marginals, result, addons) = println("After message called!")
```
"""
const AfterMessageRuleCallback = Val{:after_message_rule_call}
