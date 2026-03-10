
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

See also: [`ReactiveMP.merge_callbacks`](@ref)
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

invoke_callback(callbacks::Ref, event, args...) = invoke_callback(callbacks[], event, args...)

"""
    MergedCallbacks{F, C}(reduce_fn, callbacks)

The result of the [`ReactiveMP.merge_callbacks`](@ref) procedure.
"""
struct MergedCallbacks{F, C}
    reduce_fn::F
    callbacks::C
end

"""
    merge_callbacks(callbacks_handlers...; reduce_fn = nothing)

This function accept an arbitrary amount of callback handlers and merges them together. 
Some callback handlers may or may not react on certain type of events.

```jldoctest
julia> handler1 = (event1 = (args...) -> println("Event 1 from handler 1"), event2 = (args...) -> println("Event 2 from handler 1"));

julia> handler2 = (event1 = (args...) -> println("Event 1 from handler 2"),);

julia> merged_handler = ReactiveMP.merge_callbacks(handler1, handler2);

julia> ReactiveMP.invoke_callback(merged_handler, Val(:event1));
Event 1 from handler 1
Event 1 from handler 2

julia> ReactiveMP.invoke_callback(merged_handler, Val(:event2));
Event 2 from handler 1
```

If `reduce_fn` is not `nothing`, the result of all the callbacks will be reduced
with the provided reduce function.

```jldoctest
julia> callback_handler1 = (event1 = (a, b) -> a + b,);

julia> callback_handler2 = (event1 = (a, b) -> a * b,);

julia> merged_handler = ReactiveMP.merge_callbacks(callback_handler1, callback_handler2);

julia> ReactiveMP.invoke_callback(merged_handler, Val(:event1), 2, 3)
(5, 6)

julia> merged_handler_with_reduce = ReactiveMP.merge_callbacks(callback_handler1, callback_handler2; reduce_fn = +);

julia> ReactiveMP.invoke_callback(merged_handler_with_reduce, Val(:event1), 2, 3)
11
```

The `reduce_fn` can also be a `NamedTuple` that sets different reduce functions for 
different events.

```jldoctest
julia> callback_handler1 = (event1 = (a, b) -> a + b, event2 = (a, b) -> a - b);

julia> callback_handler2 = (event1 = (a, b) -> a * b, event2 = (a, b) -> a / b);

julia> merged_handler = ReactiveMP.merge_callbacks(callback_handler1, callback_handler2; reduce_fn = (
           event1 = +,
           event2 = *
       ));

julia> ReactiveMP.invoke_callback(merged_handler, Val(:event1), 4, 5)
29

julia> ReactiveMP.invoke_callback(merged_handler, Val(:event2), 5, 5)
0
```

See also: [`ReactiveMP.invoke_callback`](@ref)
"""
function merge_callbacks(callback_handlers...; reduce_fn = nothing)
    return MergedCallbacks(reduce_fn, callback_handlers)
end

"""
    invoke_callback(merged::MergedCallbacks, event, args...)

A specialized version of [`ReactiveMP.invoke_callback`](@ref) for [`ReactiveMP.MergedCallbacks`](@ref). 
Calls the provided callbacks in order and uses the provided reduce function to 
reduce the collection of results into a single one.
"""
function invoke_callback(merged::MergedCallbacks, event, args...)
    result = map(merged.callbacks) do callback
        invoke_callback(callback, event, args...)
    end
    return merged_callback_reduce_result(merged.reduce_fn, event, result)
end

merged_callback_reduce_result(::Nothing, _, result) = result
merged_callback_reduce_result(reduce_fn::F, _, result) where {F} = reduce(reduce_fn, result)
# If `reduce_fn` is a NamedTuple, then we choose a specific function for a specific event from this tuple
merged_callback_reduce_result(reduce_fn::NamedTuple{K}, event::Val{E}, result) where {K, E} = merged_callback_reduce_result(get(reduce_fn, E, nothing), event, result)

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
