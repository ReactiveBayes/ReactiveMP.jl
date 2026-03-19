
"""
    invoke_callback(callbacks, event::Val{E}, data)

Invokes the callback handler `callbacks` for the given `event` with a structured `data`
object. Does nothing if `callbacks` is `nothing`.

The `event` is a `Val{:symbol}` that identifies the event type. The `data` argument is a
plain struct (defined with `Base.@kwdef`) that holds all relevant information as named
fields, so handlers can write `data.field` instead of indexing into positional arguments.

Custom callback handlers should implement `invoke_callback` for specific event types:

```jldoctest
julia> struct MyCallbackHandler end

julia> ReactiveMP.invoke_callback(::MyCallbackHandler, ::Val{:my_event}, data) = print("x = \$(data.x)")

julia> Base.@kwdef struct MyEventData; x::Int; end

julia> ReactiveMP.invoke_callback(MyCallbackHandler(), Val{:my_event}(), MyEventData(x = 42))
x = 42
```

See also: [`ReactiveMP.merge_callbacks`](@ref)
"""
function invoke_callback(::Nothing, ::Val, data)
    return nothing
end

"""
    invoke_callback(callbacks::NamedTuple, event::Val{E}, data)

The `callbacks` can be a `NamedTuple` with fields corresponding to event names.
The field name must match the symbol `E` of the `Val{E}` event. The handler function
receives the `data` struct as its only argument.

```jldoctest
julia> Base.@kwdef struct MyEventData; x::Int; y::Int; end

julia> callbacks = (my_event = (data) -> data.x + data.y,);

julia> ReactiveMP.invoke_callback(callbacks, Val{:my_event}(), MyEventData(x = 1, y = 2))
3

julia> ReactiveMP.invoke_callback(callbacks, Val{:my_event}(), MyEventData(x = 10, y = 20))
30
```

If the `NamedTuple` does not have a field matching the event name, the event is ignored.
"""
function invoke_callback(
    callbacks::NamedTuple{K}, ::Val{E}, data
) where {K, E}
    if E in K
        return callbacks[E](data)
    end
    return nothing
end

"""
    invoke_callback(callbacks::Dict{Symbol}, event::Val{E}, data)

The `callbacks` can be a `Dict{Symbol, Any}` with keys corresponding to event names.
Works the same as the `NamedTuple` variant, but allows dynamic construction of callback
handlers at runtime. The handler function receives the `data` struct.

```jldoctest
julia> Base.@kwdef struct MyEventData; x::Int; y::Int; end

julia> callbacks = Dict(:my_event => (data) -> data.x + data.y);

julia> ReactiveMP.invoke_callback(callbacks, Val{:my_event}(), MyEventData(x = 1, y = 2))
3

julia> ReactiveMP.invoke_callback(callbacks, Val{:my_event}(), MyEventData(x = 10, y = 20))
30
```

If the `Dict` does not have a key matching the event name, the event is ignored.
"""
function invoke_callback(callbacks::Dict{Symbol}, ::Val{E}, data) where {E}
    if haskey(callbacks, E)
        return callbacks[E](data)
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

Accepts an arbitrary number of callback handlers and merges them together.
Each handler may react to different events independently.

```jldoctest
julia> handler1 = (event1 = (data) -> println("Event 1: \$(data.x)"), event2 = (data) -> println("Event 2: \$(data.x)"));

julia> handler2 = (event1 = (data) -> println("Event 1 again: \$(data.x)"),);

julia> merged = ReactiveMP.merge_callbacks(handler1, handler2);

julia> Base.@kwdef struct Ev1; x::Int; end

julia> Base.@kwdef struct Ev2; x::Int; end

julia> ReactiveMP.invoke_callback(merged, Val{:event1}(), Ev1(x = 42));
Event 1: 42
Event 1 again: 42

julia> ReactiveMP.invoke_callback(merged, Val{:event2}(), Ev2(x = 7));
Event 2: 7
```

If `reduce_fn` is not `nothing`, the results of all handlers for an event are reduced
with the provided function.

```jldoctest
julia> Base.@kwdef struct SumData{A, B}; a::A; b::B; end

julia> h1 = (sum_event = (data) -> data.a + data.b,);

julia> h2 = (sum_event = (data) -> data.a * data.b,);

julia> merged = ReactiveMP.merge_callbacks(h1, h2; reduce_fn = +);

julia> ReactiveMP.invoke_callback(merged, Val{:sum_event}(), SumData(a = 2, b = 3))
11
```

The `reduce_fn` can also be a `NamedTuple` to set different reduce functions per event.

```jldoctest
julia> Base.@kwdef struct EvA{A, B}; a::A; b::B; end

julia> Base.@kwdef struct EvB{A, B}; a::A; b::B; end

julia> h1 = (event_a = (d) -> d.a + d.b, event_b = (d) -> d.a - d.b);

julia> h2 = (event_a = (d) -> d.a * d.b, event_b = (d) -> d.a / d.b);

julia> merged = ReactiveMP.merge_callbacks(h1, h2; reduce_fn = (event_a = +, event_b = *));

julia> ReactiveMP.invoke_callback(merged, Val{:event_a}(), EvA(a = 4, b = 5))
29

julia> ReactiveMP.invoke_callback(merged, Val{:event_b}(), EvB(a = 5.0, b = 5.0))
0.0
```

See also: [`ReactiveMP.invoke_callback`](@ref)
"""
function merge_callbacks(callback_handlers...; reduce_fn = nothing)
    return MergedCallbacks(reduce_fn, callback_handlers)
end

"""
    invoke_callback(merged::MergedCallbacks, event::Val{E}, data)

A specialized version of [`ReactiveMP.invoke_callback`](@ref) for [`ReactiveMP.MergedCallbacks`](@ref).
Calls each contained callback handler in order and optionally reduces the results with the
provided reduce function.
"""
function invoke_callback(merged::MergedCallbacks, event::Val{E}, data) where {E}
    result = map(merged.callbacks) do callback
        invoke_callback(callback, event, data)
    end
    return merged_callback_reduce_result(merged.reduce_fn, event, result)
end

merged_callback_reduce_result(::Nothing, _, result) = result
merged_callback_reduce_result(reduce_fn::F, _, result) where {F} = reduce(
    reduce_fn, result
)
# If `reduce_fn` is a NamedTuple, then we choose a specific function for a specific event from this tuple
merged_callback_reduce_result(reduce_fn::NamedTuple{K}, event::Val{E}, result) where {K, E} = merged_callback_reduce_result(
    get(reduce_fn, E, nothing), event, result
)

# All defined event data types go here, so it's easier to document them all in one place.

"""
    BeforeMessageRuleCallback # Val{:before_message_rule_call}

Alias for `Val{:before_message_rule_call}`. Fires right **before** a message rule is
computed. The callback handler receives a [`ReactiveMP.BeforeMessageRuleCallbackData`](@ref)
as its `data` argument.

```jldoctest
julia> import ReactiveMP: BeforeMessageRuleCallback, BeforeMessageRuleCallbackData

julia> struct MyHandler end

julia> ReactiveMP.invoke_callback(::MyHandler, ::BeforeMessageRuleCallback, data::BeforeMessageRuleCallbackData) =
           println("Before rule, node = \$(ReactiveMP.message_mapping_fform(data.mapping))")
```

See also: [`ReactiveMP.AfterMessageRuleCallback`](@ref), [`ReactiveMP.BeforeMessageRuleCallbackData`](@ref)
"""
const BeforeMessageRuleCallback = Val{:before_message_rule_call}

"""
    AfterMessageRuleCallback # Val{:after_message_rule_call}

Alias for `Val{:after_message_rule_call}`. Fires right **after** a message rule is
computed. The callback handler receives an [`ReactiveMP.AfterMessageRuleCallbackData`](@ref)
as its `data` argument.

See also: [`ReactiveMP.BeforeMessageRuleCallback`](@ref), [`ReactiveMP.AfterMessageRuleCallbackData`](@ref)
"""
const AfterMessageRuleCallback = Val{:after_message_rule_call}

"""
    BeforeProductOfTwoMessages # Val{:before_product_of_two_messages}

Alias for `Val{:before_product_of_two_messages}`. Fires right **before** computing the
product of two messages. The callback handler receives a
[`ReactiveMP.BeforeProductOfTwoMessagesData`](@ref) as its `data` argument.

See also: [`ReactiveMP.AfterProductOfTwoMessages`](@ref), [`ReactiveMP.BeforeProductOfTwoMessagesData`](@ref)
"""
const BeforeProductOfTwoMessages = Val{:before_product_of_two_messages}

"""
    AfterProductOfTwoMessages # Val{:after_product_of_two_messages}

Alias for `Val{:after_product_of_two_messages}`. Fires right **after** computing the
product of two messages. The callback handler receives an
[`ReactiveMP.AfterProductOfTwoMessagesData`](@ref) as its `data` argument.

See also: [`ReactiveMP.BeforeProductOfTwoMessages`](@ref), [`ReactiveMP.AfterProductOfTwoMessagesData`](@ref)
"""
const AfterProductOfTwoMessages = Val{:after_product_of_two_messages}

"""
    BeforeProductOfMessages # Val{:before_product_of_messages}

Alias for `Val{:before_product_of_messages}`. Fires right **before** computing the product
of a collection of messages. The callback handler receives a
[`ReactiveMP.BeforeProductOfMessagesData`](@ref) as its `data` argument.

See also: [`ReactiveMP.AfterProductOfMessages`](@ref), [`ReactiveMP.BeforeProductOfMessagesData`](@ref)
"""
const BeforeProductOfMessages = Val{:before_product_of_messages}

"""
    AfterProductOfMessages # Val{:after_product_of_messages}

Alias for `Val{:after_product_of_messages}`. Fires right **after** computing the product
of a collection of messages. The callback handler receives an
[`ReactiveMP.AfterProductOfMessagesData`](@ref) as its `data` argument.

See also: [`ReactiveMP.BeforeProductOfMessages`](@ref), [`ReactiveMP.AfterProductOfMessagesData`](@ref)
"""
const AfterProductOfMessages = Val{:after_product_of_messages}

"""
    BeforeFormConstraintApplied # Val{:before_form_constraint_applied}

Alias for `Val{:before_form_constraint_applied}`. Fires right **before** applying a form
constraint via [`ReactiveMP.constrain_form`](@ref). The callback handler receives a
[`ReactiveMP.BeforeFormConstraintAppliedData`](@ref) as its `data` argument.

See also: [`ReactiveMP.AfterFormConstraintApplied`](@ref), [`ReactiveMP.BeforeFormConstraintAppliedData`](@ref)
"""
const BeforeFormConstraintApplied = Val{:before_form_constraint_applied}

"""
    AfterFormConstraintApplied # Val{:after_form_constraint_applied}

Alias for `Val{:after_form_constraint_applied}`. Fires right **after** applying a form
constraint via [`ReactiveMP.constrain_form`](@ref). The callback handler receives an
[`ReactiveMP.AfterFormConstraintAppliedData`](@ref) as its `data` argument.

See also: [`ReactiveMP.BeforeFormConstraintApplied`](@ref), [`ReactiveMP.AfterFormConstraintAppliedData`](@ref)
"""
const AfterFormConstraintApplied = Val{:after_form_constraint_applied}

"""
    BeforeMarginalComputation # Val{:before_marginal_computation}

Alias for `Val{:before_marginal_computation}`. Fires right **before** computing the
marginal for a [`ReactiveMP.RandomVariable`](@ref) from its incoming messages. The callback
handler receives a [`ReactiveMP.BeforeMarginalComputationData`](@ref) as its `data`
argument.

See also: [`ReactiveMP.AfterMarginalComputation`](@ref), [`ReactiveMP.BeforeMarginalComputationData`](@ref)
"""
const BeforeMarginalComputation = Val{:before_marginal_computation}

"""
    AfterMarginalComputation # Val{:after_marginal_computation}

Alias for `Val{:after_marginal_computation}`. Fires right **after** computing the marginal
for a [`ReactiveMP.RandomVariable`](@ref) from its incoming messages. The callback handler
receives an [`ReactiveMP.AfterMarginalComputationData`](@ref) as its `data` argument.

See also: [`ReactiveMP.BeforeMarginalComputation`](@ref), [`ReactiveMP.AfterMarginalComputationData`](@ref)
"""
const AfterMarginalComputation = Val{:after_marginal_computation}

# Structured data types for each event.
# These are plain structs (no abstract supertype required) defined with Base.@kwdef
# so they support keyword constructors and named field access.

"""
    BeforeMessageRuleCallbackData

Structured data passed to callbacks for the [`ReactiveMP.BeforeMessageRuleCallback`](@ref)
event. Access event information via named fields on the `data` argument.

## Fields
- `mapping` — the [`ReactiveMP.MessageMapping`](@ref) containing node type, tags, meta, etc.
- `messages` — the incoming messages (typically a `Tuple`, or `nothing`)
- `marginals` — the incoming marginals (typically a `Tuple`, or `nothing`)

See also: [`ReactiveMP.AfterMessageRuleCallbackData`](@ref), [`ReactiveMP.BeforeMessageRuleCallback`](@ref)
"""
Base.@kwdef struct BeforeMessageRuleCallbackData{M, MS, MA}
    mapping::M
    messages::MS
    marginals::MA
end

"""
    AfterMessageRuleCallbackData

Structured data passed to callbacks for the [`ReactiveMP.AfterMessageRuleCallback`](@ref)
event.

## Fields
- `mapping` — the [`ReactiveMP.MessageMapping`](@ref) containing node type, tags, meta, etc.
- `messages` — the incoming messages (typically a `Tuple`, or `nothing`)
- `marginals` — the incoming marginals (typically a `Tuple`, or `nothing`)
- `result` — the result of the rule invocation, can be any type
- `addons` — the computed addons after rule execution (can be `nothing`)

See also: [`ReactiveMP.BeforeMessageRuleCallbackData`](@ref), [`ReactiveMP.AfterMessageRuleCallback`](@ref)
"""
Base.@kwdef struct AfterMessageRuleCallbackData{M, MS, MA, R, A}
    mapping::M
    messages::MS
    marginals::MA
    result::R
    addons::A
end

"""
    BeforeProductOfTwoMessagesData

Structured data passed to callbacks for the
[`ReactiveMP.BeforeProductOfTwoMessages`](@ref) event.

## Fields
- `variable` — the [`ReactiveMP.AbstractVariable`](@ref) for which the product is computed
- `context` — the [`ReactiveMP.MessageProductContext`](@ref)
- `left` — the left-hand [`ReactiveMP.Message`](@ref)
- `right` — the right-hand [`ReactiveMP.Message`](@ref)

See also: [`ReactiveMP.AfterProductOfTwoMessagesData`](@ref), [`ReactiveMP.BeforeProductOfTwoMessages`](@ref)
"""
Base.@kwdef struct BeforeProductOfTwoMessagesData{V, C, L, R}
    variable::V
    context::C
    left::L
    right::R
end

"""
    AfterProductOfTwoMessagesData

Structured data passed to callbacks for the
[`ReactiveMP.AfterProductOfTwoMessages`](@ref) event.

## Fields
- `variable` — the [`ReactiveMP.AbstractVariable`](@ref) for which the product is computed
- `context` — the [`ReactiveMP.MessageProductContext`](@ref)
- `left` — the left-hand [`ReactiveMP.Message`](@ref)
- `right` — the right-hand [`ReactiveMP.Message`](@ref)
- `result` — the resulting [`ReactiveMP.Message`](@ref) from the product
- `addons` — the computed addons for the result (can be `nothing`)

See also: [`ReactiveMP.BeforeProductOfTwoMessagesData`](@ref), [`ReactiveMP.AfterProductOfTwoMessages`](@ref)
"""
Base.@kwdef struct AfterProductOfTwoMessagesData{V, C, L, R, Re, A}
    variable::V
    context::C
    left::L
    right::R
    result::Re
    addons::A
end

"""
    BeforeProductOfMessagesData

Structured data passed to callbacks for the [`ReactiveMP.BeforeProductOfMessages`](@ref)
event.

## Fields
- `variable` — the [`ReactiveMP.AbstractVariable`](@ref) for which the product is computed
- `context` — the [`ReactiveMP.MessageProductContext`](@ref)
- `messages` — the collection of messages to be multiplied

See also: [`ReactiveMP.AfterProductOfMessagesData`](@ref), [`ReactiveMP.BeforeProductOfMessages`](@ref)
"""
Base.@kwdef struct BeforeProductOfMessagesData{V, C, M}
    variable::V
    context::C
    messages::M
end

"""
    AfterProductOfMessagesData

Structured data passed to callbacks for the [`ReactiveMP.AfterProductOfMessages`](@ref)
event.

## Fields
- `variable` — the [`ReactiveMP.AbstractVariable`](@ref) for which the product is computed
- `context` — the [`ReactiveMP.MessageProductContext`](@ref)
- `messages` — the original collection of messages that were multiplied
- `result` — the final [`ReactiveMP.Message`](@ref) after folding and form constraint
  application

See also: [`ReactiveMP.BeforeProductOfMessagesData`](@ref), [`ReactiveMP.AfterProductOfMessages`](@ref)
"""
Base.@kwdef struct AfterProductOfMessagesData{V, C, M, Re}
    variable::V
    context::C
    messages::M
    result::Re
end

"""
    BeforeFormConstraintAppliedData

Structured data passed to callbacks for the
[`ReactiveMP.BeforeFormConstraintApplied`](@ref) event. Fires in both
[`ReactiveMP.FormConstraintCheckEach`](@ref) and
[`ReactiveMP.FormConstraintCheckLast`](@ref) strategies.

## Fields
- `variable` — the [`ReactiveMP.AbstractVariable`](@ref)
- `context` — the [`ReactiveMP.MessageProductContext`](@ref)
- `strategy` — the form constraint check strategy (e.g.
  [`ReactiveMP.FormConstraintCheckEach`](@ref) or
  [`ReactiveMP.FormConstraintCheckLast`](@ref))
- `distribution` — the distribution about to be constrained

See also: [`ReactiveMP.AfterFormConstraintAppliedData`](@ref), [`ReactiveMP.BeforeFormConstraintApplied`](@ref)
"""
Base.@kwdef struct BeforeFormConstraintAppliedData{V, C, S, D}
    variable::V
    context::C
    strategy::S
    distribution::D
end

"""
    AfterFormConstraintAppliedData

Structured data passed to callbacks for the
[`ReactiveMP.AfterFormConstraintApplied`](@ref) event. Fires in both
[`ReactiveMP.FormConstraintCheckEach`](@ref) and
[`ReactiveMP.FormConstraintCheckLast`](@ref) strategies.

## Fields
- `variable` — the [`ReactiveMP.AbstractVariable`](@ref)
- `context` — the [`ReactiveMP.MessageProductContext`](@ref)
- `strategy` — the form constraint check strategy (e.g.
  [`ReactiveMP.FormConstraintCheckEach`](@ref) or
  [`ReactiveMP.FormConstraintCheckLast`](@ref))
- `distribution` — the distribution **before** the constraint was applied
- `result` — the distribution **after** the constraint was applied

See also: [`ReactiveMP.BeforeFormConstraintAppliedData`](@ref), [`ReactiveMP.AfterFormConstraintApplied`](@ref)
"""
Base.@kwdef struct AfterFormConstraintAppliedData{V, C, S, D, R}
    variable::V
    context::C
    strategy::S
    distribution::D
    result::R
end

"""
    BeforeMarginalComputationData

Structured data passed to callbacks for the [`ReactiveMP.BeforeMarginalComputation`](@ref)
event.

## Fields
- `variable` — the [`ReactiveMP.RandomVariable`](@ref)
- `context` — the [`ReactiveMP.MessageProductContext`](@ref)
- `messages` — the collection of incoming messages used to compute the marginal

See also: [`ReactiveMP.AfterMarginalComputationData`](@ref), [`ReactiveMP.BeforeMarginalComputation`](@ref)
"""
Base.@kwdef struct BeforeMarginalComputationData{V, C, M}
    variable::V
    context::C
    messages::M
end

"""
    AfterMarginalComputationData

Structured data passed to callbacks for the [`ReactiveMP.AfterMarginalComputation`](@ref)
event.

## Fields
- `variable` — the [`ReactiveMP.RandomVariable`](@ref)
- `context` — the [`ReactiveMP.MessageProductContext`](@ref)
- `messages` — the collection of incoming messages used to compute the marginal
- `result` — the computed marginal

See also: [`ReactiveMP.BeforeMarginalComputationData`](@ref), [`ReactiveMP.AfterMarginalComputation`](@ref)
"""
Base.@kwdef struct AfterMarginalComputationData{V, C, M, R}
    variable::V
    context::C
    messages::M
    result::R
end
