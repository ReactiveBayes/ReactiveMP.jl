
"""
    Event{E}

Abstract supertype for all callback events in the reactive message passing procedure.
`E` is a `Symbol` that identifies the event, e.g. `Event{:before_message_rule_call}`.

Concrete event types should subtype `Event{:event_name}` and carry the relevant data as fields.
The naming convention is that for an event `:event_name`, the corresponding struct is called `EventNameEvent`.

See also: [`ReactiveMP.invoke_callback`](@ref)
"""
abstract type Event{E} end

"""
    event_name(::Type{<:Event{E}}) where {E}

Returns the event name symbol `E` from an `Event{E}` type or any of its subtypes.
"""
event_name(::Type{<:Event{E}}) where {E} = E

"""
    invoke_callback(callbacks, event::Event)

Custom callbacks handlers should implement `invoke_callback` in order to listen to events
during the reactive message passing procedure.
Each event is a subtype of [`ReactiveMP.Event{E}`](@ref) that carries the relevant data as fields.
Does nothing if `callbacks` is `nothing`.

```jldoctest
julia> struct MyEvent <: ReactiveMP.Event{:my_event}
           value::Int
       end;

julia> struct MyCustomCallbackHandler end;

julia> ReactiveMP.invoke_callback(::MyCustomCallbackHandler, event::MyEvent) = print("Event value: \$(event.value)");
```

See also: [`ReactiveMP.Event`](@ref), [`ReactiveMP.merge_callbacks`](@ref)
"""
function invoke_callback(callbacks::Nothing, ::Event)
    return nothing
end

"""
    invoke_callback(callbacks::NamedTuple, event::Event{E})

The `callbacks` can also be a `NamedTuple` with fields corresponding to event names.
Each callback function receives the event object itself.

```jldoctest
julia> struct SumEvent <: ReactiveMP.Event{:sum_event}
           a::Int; b::Int
       end;

julia> callbacks = (sum_event = (event) -> event.a + event.b,);

julia> ReactiveMP.invoke_callback(callbacks, SumEvent(1, 2))
3

julia> ReactiveMP.invoke_callback(callbacks, SumEvent(3, 4))
7
```

If the `NamedTuple` does not have a field corresponding to the event name, the event will be ignored.
"""
function invoke_callback(
    callbacks::NamedTuple{K}, event::Event{E}
) where {K, E}
    if E in K
        return callbacks[E](event)
    end
    return nothing
end

"""
    invoke_callback(callbacks::Dict{Symbol}, event::Event{E})

The `callbacks` can also be a `Dict{Symbol, Any}` with keys corresponding to event names.
Works the same as the `NamedTuple` variant, but allows dynamic construction of callback handlers at runtime.
Each callback function receives the event object itself.

```jldoctest
julia> struct SumEvent <: ReactiveMP.Event{:sum_event}
           a::Int; b::Int
       end;

julia> callbacks = Dict(:sum_event => (event) -> event.a + event.b);

julia> ReactiveMP.invoke_callback(callbacks, SumEvent(1, 2))
3

julia> ReactiveMP.invoke_callback(callbacks, SumEvent(3, 4))
7
```

If the `Dict` does not have a key corresponding to the event name, the event will be ignored.
"""
function invoke_callback(callbacks::Dict{Symbol}, event::Event{E}) where {E}
    if haskey(callbacks, E)
        return callbacks[E](event)
    end
    return nothing
end

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

If `reduce_fn` is not `nothing`, the result of all the callbacks will be reduced
with the provided reduce function.

The `reduce_fn` can also be a `NamedTuple` that sets different reduce functions for
different events.

See also: [`ReactiveMP.invoke_callback`](@ref)
"""
function merge_callbacks(callback_handlers...; reduce_fn = nothing)
    return MergedCallbacks(reduce_fn, callback_handlers)
end

"""
    invoke_callback(merged::MergedCallbacks, event::Event)

A specialized version of [`ReactiveMP.invoke_callback`](@ref) for [`ReactiveMP.MergedCallbacks`](@ref).
Calls the provided callbacks in order and uses the provided reduce function to
reduce the collection of results into a single one.
"""
function invoke_callback(merged::MergedCallbacks, event::Event)
    result = map(merged.callbacks) do callback
        invoke_callback(callback, event)
    end
    return merged_callback_reduce_result(merged.reduce_fn, event, result)
end

merged_callback_reduce_result(::Nothing, _, result) = result
merged_callback_reduce_result(reduce_fn::F, _, result) where {F} = reduce(
    reduce_fn, result
)
# If `reduce_fn` is a NamedTuple, then we choose a specific function for a specific event from this tuple
merged_callback_reduce_result(reduce_fn::NamedTuple{K}, event::Event{E}, result) where {K, E} = merged_callback_reduce_result(
    get(reduce_fn, E, nothing), event, result
)

# All defined events go here, so its easier to document them all in one place

"""
    BeforeMessageRuleCallEvent{M, Ms, Mr} <: Event{:before_message_rule_call}

This event fires right before computing the message and calling the corresponding rule.

# Fields
- `mapping`: of type [`ReactiveMP.MessageMapping`](@ref), contains information about the node type, etc
- `messages`: typically of type `Tuple` if present, `nothing` otherwise
- `marginals`: typically of type `Tuple` if present, `nothing` otherwise

See also: [`ReactiveMP.invoke_callback`](@ref), [`ReactiveMP.AfterMessageRuleCallEvent`](@ref)
"""
struct BeforeMessageRuleCallEvent{M, Ms, Mr} <: Event{:before_message_rule_call}
    mapping::M
    messages::Ms
    marginals::Mr
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

See also: [`ReactiveMP.invoke_callback`](@ref), [`ReactiveMP.BeforeMessageRuleCallEvent`](@ref)
"""
struct AfterMessageRuleCallEvent{M, Ms, Mr, R, A} <: Event{:after_message_rule_call}
    mapping::M
    messages::Ms
    marginals::Mr
    result::R
    addons::A
end

"""
    BeforeProductOfTwoMessagesEvent{V, C, L, R} <: Event{:before_product_of_two_messages}

This event fires right before computing the product of two messages.

# Fields
- `variable`: of type [`ReactiveMP.AbstractVariable`](@ref)
- `context`: of type [`ReactiveMP.MessageProductContext`](@ref)
- `left`: of type [`ReactiveMP.Message`](@ref), the left-hand side message in the product
- `right`: of type [`ReactiveMP.Message`](@ref), the right-hand side message in the product

See also: [`ReactiveMP.invoke_callback`](@ref), [`ReactiveMP.AfterProductOfTwoMessagesEvent`](@ref)
"""
struct BeforeProductOfTwoMessagesEvent{V, C, L, R} <: Event{:before_product_of_two_messages}
    variable::V
    context::C
    left::L
    right::R
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

See also: [`ReactiveMP.invoke_callback`](@ref), [`ReactiveMP.BeforeProductOfTwoMessagesEvent`](@ref)
"""
struct AfterProductOfTwoMessagesEvent{V, C, L, R, Rs, A} <: Event{:after_product_of_two_messages}
    variable::V
    context::C
    left::L
    right::R
    result::Rs
    addons::A
end

"""
    BeforeProductOfMessagesEvent{V, C, Ms} <: Event{:before_product_of_messages}

This event fires right before computing the product of a collection of messages
(i.e. at the beginning of [`ReactiveMP.compute_product_of_messages`](@ref)).

# Fields
- `variable`: of type [`ReactiveMP.AbstractVariable`](@ref)
- `context`: of type [`ReactiveMP.MessageProductContext`](@ref)
- `messages`: the collection of messages to be multiplied

See also: [`ReactiveMP.invoke_callback`](@ref), [`ReactiveMP.AfterProductOfMessagesEvent`](@ref)
"""
struct BeforeProductOfMessagesEvent{V, C, Ms} <: Event{:before_product_of_messages}
    variable::V
    context::C
    messages::Ms
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

See also: [`ReactiveMP.invoke_callback`](@ref), [`ReactiveMP.BeforeProductOfMessagesEvent`](@ref)
"""
struct AfterProductOfMessagesEvent{V, C, Ms, R} <: Event{:after_product_of_messages}
    variable::V
    context::C
    messages::Ms
    result::R
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

See also: [`ReactiveMP.invoke_callback`](@ref), [`ReactiveMP.AfterFormConstraintAppliedEvent`](@ref)
"""
struct BeforeFormConstraintAppliedEvent{V, C, S, D} <: Event{:before_form_constraint_applied}
    variable::V
    context::C
    strategy::S
    distribution::D
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

See also: [`ReactiveMP.invoke_callback`](@ref), [`ReactiveMP.BeforeFormConstraintAppliedEvent`](@ref)
"""
struct AfterFormConstraintAppliedEvent{V, C, S, D, R} <: Event{:after_form_constraint_applied}
    variable::V
    context::C
    strategy::S
    distribution::D
    result::R
end

"""
    BeforeMarginalComputationEvent{V, C, Ms} <: Event{:before_marginal_computation}

This event fires right before computing the marginal for a [`ReactiveMP.RandomVariable`](@ref) from its incoming messages.

# Fields
- `variable`: of type [`ReactiveMP.RandomVariable`](@ref)
- `context`: of type [`ReactiveMP.MessageProductContext`](@ref)
- `messages`: the collection of incoming messages used to compute the marginal

See also: [`ReactiveMP.invoke_callback`](@ref), [`ReactiveMP.AfterMarginalComputationEvent`](@ref)
"""
struct BeforeMarginalComputationEvent{V, C, Ms} <: Event{:before_marginal_computation}
    variable::V
    context::C
    messages::Ms
end

"""
    AfterMarginalComputationEvent{V, C, Ms, R} <: Event{:after_marginal_computation}

This event fires right after computing the marginal for a [`ReactiveMP.RandomVariable`](@ref) from its incoming messages.

# Fields
- `variable`: of type [`ReactiveMP.RandomVariable`](@ref)
- `context`: of type [`ReactiveMP.MessageProductContext`](@ref)
- `messages`: the collection of incoming messages used to compute the marginal
- `result`: the computed marginal

See also: [`ReactiveMP.invoke_callback`](@ref), [`ReactiveMP.BeforeMarginalComputationEvent`](@ref)
"""
struct AfterMarginalComputationEvent{V, C, Ms, R} <: Event{:after_marginal_computation}
    variable::V
    context::C
    messages::Ms
    result::R
end
