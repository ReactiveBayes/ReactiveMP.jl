using UUIDs

"""
    Event{E}

Abstract supertype for all callback events in the reactive message passing procedure.
`E` is a `Symbol` that identifies the event, e.g. `Event{:before_message_rule_call}`.

Concrete event types should subtype `Event{:event_name}` and carry the relevant data as fields.
The naming convention is that for an event `:event_name`, the corresponding struct is called `EventNameEvent`.

See also: [`ReactiveMP.invoke_callback`](@ref), [`ReactiveMP.handle_event`](@ref)
"""
abstract type Event{E} end

"""
    event_name(::Type{<:Event{E}}) where {E}
    event_name(event::Event)

Returns the event name symbol `E` from an `Event{E}` type, subtype, or instance.
"""
event_name(::Type{<:Event{E}}) where {E} = E
event_name(event::Event) = event_name(typeof(event))

"""
    handle_event(handler, event::Event)

Custom callback handlers should implement `handle_event` to listen to events
during the reactive message passing procedure.
Each event is a subtype of [`ReactiveMP.Event{E}`](@ref) that carries the relevant data as fields.
The return value of `handle_event` is ignored. To communicate state changes, use mutable event fields.

```jldoctest
julia> struct MyEvent <: ReactiveMP.Event{:my_event}
           value::Int
       end;

julia> struct MyCustomCallbackHandler end;

julia> ReactiveMP.handle_event(::MyCustomCallbackHandler, event::MyEvent) = print("Event value: \$(event.value)");
```

See also: [`ReactiveMP.Event`](@ref), [`ReactiveMP.invoke_callback`](@ref), [`ReactiveMP.merge_callbacks`](@ref)
"""
function handle_event end

"""
    invoke_callback(callbacks, event::Event)

Invokes the callback handler(s) for the given event and returns the `event` itself.
Internally dispatches to [`ReactiveMP.handle_event`](@ref) for each handler.
Does nothing and returns the event if `callbacks` is `nothing`.

See also: [`ReactiveMP.handle_event`](@ref), [`ReactiveMP.Event`](@ref), [`ReactiveMP.merge_callbacks`](@ref)
"""
function invoke_callback(callbacks::Nothing, event::Event)
    return event
end

"""
    invoke_callback(callbacks::NamedTuple, event::Event{E})

The `callbacks` can also be a `NamedTuple` with fields corresponding to event names.
Each callback function receives the event object itself. The return value of the callback is ignored.

```jldoctest
julia> mutable struct CountEvent <: ReactiveMP.Event{:count_event}
           count::Int
       end;

julia> callbacks = (count_event = (event) -> event.count += 1,);

julia> event = CountEvent(0);

julia> ReactiveMP.invoke_callback(callbacks, event);

julia> event.count
1
```

If the `NamedTuple` does not have a field corresponding to the event name, the event will be ignored.
"""
function invoke_callback(callbacks::NamedTuple{K}, event::Event{E}) where {K, E}
    if E in K
        callbacks[E](event)
    end
    return event
end

"""
    invoke_callback(callbacks::Dict{Symbol}, event::Event{E})

The `callbacks` can also be a `Dict{Symbol, Any}` with keys corresponding to event names.
Works the same as the `NamedTuple` variant, but allows dynamic construction of callback handlers at runtime.
Each callback function receives the event object itself. The return value of the callback is ignored.

If the `Dict` does not have a key corresponding to the event name, the event will be ignored.
"""
function invoke_callback(callbacks::Dict{Symbol}, event::Event{E}) where {E}
    if haskey(callbacks, E)
        callbacks[E](event)
    end
    return event
end

"""
    invoke_callback(handler, event::Event)

Fallback for custom callback handlers. Delegates to [`ReactiveMP.handle_event`](@ref) and returns the `event`.
Custom handlers should implement `handle_event(handler, event)` rather than `invoke_callback`.
"""
function invoke_callback(handler, event::Event)
    handle_event(handler, event)
    return event
end

"""
    MergedCallbacks{C}(callbacks)

The result of the [`ReactiveMP.merge_callbacks`](@ref) procedure.
"""
struct MergedCallbacks{C}
    callbacks::C
end

"""
    merge_callbacks(callbacks_handlers...)

This function accepts an arbitrary amount of callback handlers and merges them together.
Some callback handlers may or may not react on certain types of events.

```jldoctest
julia> struct PrintEvent <: ReactiveMP.Event{:print_event}
           label::String
       end;

julia> handler1 = (print_event = (event) -> println("Handler 1: ", event.label),);

julia> handler2 = (print_event = (event) -> println("Handler 2: ", event.label),);

julia> merged_handler = ReactiveMP.merge_callbacks(handler1, handler2);

julia> ReactiveMP.invoke_callback(merged_handler, PrintEvent("hello"));
Handler 1: hello
Handler 2: hello
```

See also: [`ReactiveMP.invoke_callback`](@ref), [`ReactiveMP.handle_event`](@ref)
"""
function merge_callbacks(callback_handlers...)
    return MergedCallbacks(callback_handlers)
end

"""
    invoke_callback(merged::MergedCallbacks, event::Event)

A specialized version of [`ReactiveMP.invoke_callback`](@ref) for [`ReactiveMP.MergedCallbacks`](@ref).
Calls the provided callbacks in order. Returns the event after all handlers have been invoked.
"""
function invoke_callback(merged::MergedCallbacks, event::Event)
    for callback in merged.callbacks
        invoke_callback(callback, event)
    end
    return event
end

"""
    generate_span_id(callbacks)

Generates a unique identifier used for "before" and "after" events (see for example [`BeforeMessageRuleCallEvent`](@ref) and [`AfterMessageRuleCallEvent]`](@ref)). If callbacks are not set (e.g. `callbacks` is `nothing`), returns `nothing`.

The current implementation uses `UUIDs.uuid4` to generate span IDs, but that may change in the future.
"""
function generate_span_id end

function generate_span_id(::Nothing)
    return nothing
end

function generate_span_id(callbacks)
    return uuid4()
end

# All defined events go here, so its easier to document them all in one place

"""
    BeforeMessageRuleCallEvent{M, Ms, Mr} <: Event{:before_message_rule_call}

This event fires right before computing the message and calling the corresponding rule.

# Fields
- `mapping`: of type [`ReactiveMP.MessageMapping`](@ref), contains information about the node type, etc
- `messages`: typically of type `Tuple` if present, `nothing` otherwise
- `marginals`: typically of type `Tuple` if present, `nothing` otherwise
- `span_id`: an id shared with the corresponding [`ReactiveMP.AfterMessageRuleCallEvent`](@ref)

See also: [`ReactiveMP.invoke_callback`](@ref), [`ReactiveMP.AfterMessageRuleCallEvent`](@ref), [`ReactiveMP.generate_span_id`](@ref)
"""
struct BeforeMessageRuleCallEvent{M, Ms, Mr, S} <:
       Event{:before_message_rule_call}
    mapping::M
    messages::Ms
    marginals::Mr
    span_id::S
end

"""
    AfterMessageRuleCallEvent{M, Ms, Mr, R, A} <: Event{:after_message_rule_call}

This event fires right after computing the message and calling the corresponding rule.

# Fields
- `mapping`: of type [`ReactiveMP.MessageMapping`](@ref), contains information about the node type, etc
- `messages`: typically of type `Tuple` if present, `nothing` otherwise
- `marginals`: typically of type `Tuple` if present, `nothing` otherwise
- `result`: the result of the rule invocation (or `rulefallback`), can be any type
- `addons`: the result of the addons invocation, if present, can be any type
- `span_id`: an id shared with the corresponding [`ReactiveMP.BeforeMessageRuleCallEvent`](@ref)

See also: [`ReactiveMP.invoke_callback`](@ref), [`ReactiveMP.BeforeMessageRuleCallEvent`](@ref), [`ReactiveMP.generate_span_id`](@ref)
"""
struct AfterMessageRuleCallEvent{M, Ms, Mr, R, A, S} <:
       Event{:after_message_rule_call}
    mapping::M
    messages::Ms
    marginals::Mr
    result::R
    addons::A
    span_id::S
end

"""
    BeforeProductOfTwoMessagesEvent{V, C, L, R} <: Event{:before_product_of_two_messages}

This event fires right before computing the product of two messages.

# Fields
- `variable`: of type [`ReactiveMP.AbstractVariable`](@ref)
- `context`: of type [`ReactiveMP.MessageProductContext`](@ref)
- `left`: of type [`ReactiveMP.Message`](@ref), the left-hand side message in the product
- `right`: of type [`ReactiveMP.Message`](@ref), the right-hand side message in the product
- `span_id`: an id shared with the corresponding [`ReactiveMP.AfterProductOfTwoMessagesEvent`](@ref)

See also: [`ReactiveMP.invoke_callback`](@ref), [`ReactiveMP.AfterProductOfTwoMessagesEvent`](@ref), [`ReactiveMP.generate_span_id`](@ref)
"""
struct BeforeProductOfTwoMessagesEvent{V, C, L, R, S} <:
       Event{:before_product_of_two_messages}
    variable::V
    context::C
    left::L
    right::R
    span_id::S
end

"""
    AfterProductOfTwoMessagesEvent{V, C, L, R, Rs, A} <: Event{:after_product_of_two_messages}

This event fires right after computing the product of two messages.

# Fields
- `variable`: of type [`ReactiveMP.AbstractVariable`](@ref)
- `context`: of type [`ReactiveMP.MessageProductContext`](@ref)
- `left`: of type [`ReactiveMP.Message`](@ref), the left-hand side message in the product
- `right`: of type [`ReactiveMP.Message`](@ref), the right-hand side message in the product
- `result`: of type [`ReactiveMP.Message`](@ref), the resulting message from the product
- `addons`: the computed addons for the result (can be `nothing`)
- `span_id`: an id shared with the corresponding [`ReactiveMP.BeforeProductOfTwoMessagesEvent`](@ref)

See also: [`ReactiveMP.invoke_callback`](@ref), [`ReactiveMP.BeforeProductOfTwoMessagesEvent`](@ref), [`ReactiveMP.generate_span_id`](@ref)
"""
struct AfterProductOfTwoMessagesEvent{V, C, L, R, Rs, A, S} <:
       Event{:after_product_of_two_messages}
    variable::V
    context::C
    left::L
    right::R
    result::Rs
    addons::A
    span_id::S
end

"""
    BeforeProductOfMessagesEvent{V, C, Ms} <: Event{:before_product_of_messages}

This event fires right before computing the product of a collection of messages
(i.e. at the beginning of [`ReactiveMP.compute_product_of_messages`](@ref)).

# Fields
- `variable`: of type [`ReactiveMP.AbstractVariable`](@ref)
- `context`: of type [`ReactiveMP.MessageProductContext`](@ref)
- `messages`: the collection of messages to be multiplied
- `span_id`: an id shared with the corresponding [`ReactiveMP.AfterProductOfMessagesEvent`](@ref)

See also: [`ReactiveMP.invoke_callback`](@ref), [`ReactiveMP.AfterProductOfMessagesEvent`](@ref), [`ReactiveMP.generate_span_id`](@ref)
"""
struct BeforeProductOfMessagesEvent{V, C, Ms, S} <:
       Event{:before_product_of_messages}
    variable::V
    context::C
    messages::Ms
    span_id::S
end

"""
    AfterProductOfMessagesEvent{V, C, Ms, R} <: Event{:after_product_of_messages}

This event fires right after computing the product of a collection of messages
(i.e. at the end of [`ReactiveMP.compute_product_of_messages`](@ref)).

# Fields
- `variable`: of type [`ReactiveMP.AbstractVariable`](@ref)
- `context`: of type [`ReactiveMP.MessageProductContext`](@ref)
- `messages`: the original collection of messages that were multiplied
- `result`: of type [`ReactiveMP.Message`](@ref), the final result after folding and form constraint application
- `span_id`: an id shared with the corresponding [`ReactiveMP.BeforeProductOfMessagesEvent`](@ref)

See also: [`ReactiveMP.invoke_callback`](@ref), [`ReactiveMP.BeforeProductOfMessagesEvent`](@ref), [`ReactiveMP.generate_span_id`](@ref)
"""
struct AfterProductOfMessagesEvent{V, C, Ms, R, S} <:
       Event{:after_product_of_messages}
    variable::V
    context::C
    messages::Ms
    result::R
    span_id::S
end

"""
    BeforeFormConstraintAppliedEvent{V, C, S, D} <: Event{:before_form_constraint_applied}

This event fires right before applying the form constraint via [`ReactiveMP.constrain_form`](@ref).
Fires in both [`ReactiveMP.FormConstraintCheckEach`](@ref) and [`ReactiveMP.FormConstraintCheckLast`](@ref) strategies.

# Fields
- `variable`: of type [`ReactiveMP.AbstractVariable`](@ref)
- `context`: of type [`ReactiveMP.MessageProductContext`](@ref)
- `strategy`: the form constraint check strategy being used (e.g. [`ReactiveMP.FormConstraintCheckEach`](@ref) or [`ReactiveMP.FormConstraintCheckLast`](@ref))
- `distribution`: the distribution about to be constrained
- `span_id`: an id shared with the corresponding [`ReactiveMP.AfterFormConstraintAppliedEvent`](@ref)

See also: [`ReactiveMP.invoke_callback`](@ref), [`ReactiveMP.AfterFormConstraintAppliedEvent`](@ref), [`ReactiveMP.generate_span_id`](@ref)
"""
struct BeforeFormConstraintAppliedEvent{V, C, S, D, I} <:
       Event{:before_form_constraint_applied}
    variable::V
    context::C
    strategy::S
    distribution::D
    span_id::I
end

"""
    AfterFormConstraintAppliedEvent{V, C, S, D, R} <: Event{:after_form_constraint_applied}

This event fires right after applying the form constraint via [`ReactiveMP.constrain_form`](@ref).
Fires in both [`ReactiveMP.FormConstraintCheckEach`](@ref) and [`ReactiveMP.FormConstraintCheckLast`](@ref) strategies.

# Fields
- `variable`: of type [`ReactiveMP.AbstractVariable`](@ref)
- `context`: of type [`ReactiveMP.MessageProductContext`](@ref)
- `strategy`: the form constraint check strategy being used (e.g. [`ReactiveMP.FormConstraintCheckEach`](@ref) or [`ReactiveMP.FormConstraintCheckLast`](@ref))
- `distribution`: the distribution before the constraint was applied
- `result`: the distribution after the constraint was applied
- `span_id`: an id shared with the corresponding [`ReactiveMP.BeforeFormConstraintAppliedEvent`](@ref)

See also: [`ReactiveMP.invoke_callback`](@ref), [`ReactiveMP.BeforeFormConstraintAppliedEvent`](@ref), [`ReactiveMP.generate_span_id`](@ref)
"""
struct AfterFormConstraintAppliedEvent{V, C, S, D, R, I} <:
       Event{:after_form_constraint_applied}
    variable::V
    context::C
    strategy::S
    distribution::D
    result::R
    span_id::I
end

"""
    BeforeMarginalComputationEvent{V, C, Ms} <: Event{:before_marginal_computation}

This event fires right before computing the marginal for a [`ReactiveMP.RandomVariable`](@ref) from its incoming messages.

# Fields
- `variable`: of type [`ReactiveMP.RandomVariable`](@ref)
- `context`: of type [`ReactiveMP.MessageProductContext`](@ref)
- `messages`: the collection of incoming messages used to compute the marginal
- `span_id`: an id shared with the corresponding [`ReactiveMP.AfterMarginalComputationEvent`](@ref)

See also: [`ReactiveMP.invoke_callback`](@ref), [`ReactiveMP.AfterMarginalComputationEvent`](@ref), [`ReactiveMP.generate_span_id`](@ref)
"""
struct BeforeMarginalComputationEvent{V, C, Ms, S} <:
       Event{:before_marginal_computation}
    variable::V
    context::C
    messages::Ms
    span_id::S
end

"""
    AfterMarginalComputationEvent{V, C, Ms, R} <: Event{:after_marginal_computation}

This event fires right after computing the marginal for a [`ReactiveMP.RandomVariable`](@ref) from its incoming messages.

# Fields
- `variable`: of type [`ReactiveMP.RandomVariable`](@ref)
- `context`: of type [`ReactiveMP.MessageProductContext`](@ref)
- `messages`: the collection of incoming messages used to compute the marginal
- `span_id`: an id shared with the corresponding [`ReactiveMP.BeforeMarginalComputationEvent`](@ref)
- `result`: the computed marginal

See also: [`ReactiveMP.invoke_callback`](@ref), [`ReactiveMP.BeforeMarginalComputationEvent`](@ref), [`ReactiveMP.generate_span_id`](@ref)
"""
struct AfterMarginalComputationEvent{V, C, Ms, R, S} <:
       Event{:after_marginal_computation}
    variable::V
    context::C
    messages::Ms
    result::R
    span_id::S
end
