using UUIDs

"""
    ReactiveMP.Event{E}

The supertype of the events the engine reports to callbacks. `E` is the event's name, a
`Symbol` such as `:before_message_rule_call`, and the event for the name `:event_name` is the
struct `EventNameEvent`. An event carries what happened in its fields.

See also [`ReactiveMP.invoke_callback`](@ref), [`ReactiveMP.handle_event`](@ref).
"""
abstract type Event{E} end

"""
    ReactiveMP.event_name(::Type{<:Event{E}}) -> Symbol
    ReactiveMP.event_name(event::Event) -> Symbol

The name `E` of an event type or of an event.

# Examples

```jldoctest
julia> ReactiveMP.event_name(ReactiveMP.BeforeProductOfTwoMessagesEvent)
:before_product_of_two_messages
```
"""
event_name(::Type{<:Event{E}}) where {E} = E
event_name(event::Event) = event_name(typeof(event))

"""
    ReactiveMP.handle_event(handler, event::Event)

What a callback handler of a custom type does with `event`: a handler adds a method for each
event it reacts to, and [`ReactiveMP.invoke_callback`](@ref) calls it. Its return value is
ignored.

# Throws

- `MethodError` for an event the handler has no method for; the error's hint says how to add
  one, and that a `NamedTuple` of callbacks needs its trailing comma.

# Examples

```jldoctest
julia> struct MyEvent <: ReactiveMP.Event{:my_event}
           value::Int
       end;

julia> struct MyHandler end;

julia> ReactiveMP.handle_event(::MyHandler, event::MyEvent) = println("value: ", event.value);

julia> ReactiveMP.invoke_callback(MyHandler(), MyEvent(1));
value: 1
```

See also [`ReactiveMP.Event`](@ref), [`ReactiveMP.merge_callbacks`](@ref).
"""
function handle_event end

"""
    ReactiveMP.invoke_callback(callbacks, event::Event) -> Event

Report `event` to `callbacks`, and return the event. The engine calls it at every event, with
the callbacks given at activation; `callbacks` is one of:

- `nothing`: no callbacks, and nothing happens;
- a `NamedTuple` or a `Dict{Symbol}` keyed by event names, whose entry for the event's name, if
  any, is called with the event: `(before_message_rule_call = (event) -> …,)`;
- a [`ReactiveMP.merge_callbacks`](@ref) of several handlers, each invoked in order;
- any other handler, for which [`ReactiveMP.handle_event`](@ref) is called.

The return value of a callback is ignored.

# Examples

```jldoctest
julia> mutable struct CountEvent <: ReactiveMP.Event{:count_event}
           count::Int
       end;

julia> event = CountEvent(0);

julia> ReactiveMP.invoke_callback((count_event = (event) -> event.count += 1,), event);

julia> ReactiveMP.invoke_callback(Dict{Symbol, Any}(:count_event => (event) -> event.count += 1), event);

julia> event.count
2
```

See also [`ReactiveMP.Event`](@ref), [`ReactiveMP.handle_event`](@ref).
"""
function invoke_callback(callbacks::Nothing, event::Event)
    return event
end

function invoke_callback(callbacks::NamedTuple{K}, event::Event{E}) where {K, E}
    if E in K
        callbacks[E](event)
    end
    return event
end

function invoke_callback(callbacks::Dict{Symbol}, event::Event{E}) where {E}
    if haskey(callbacks, E)
        callbacks[E](event)
    end
    return event
end

function invoke_callback(handler, event::Event)
    handle_event(handler, event)
    return event
end

"""
    ReactiveMP.MergedCallbacks(callbacks::Tuple)

Several callback handlers as one, which [`ReactiveMP.merge_callbacks`](@ref) returns;
[`ReactiveMP.invoke_callback`](@ref) invokes each in order.
"""
struct MergedCallbacks{C}
    callbacks::C
end

"""
    ReactiveMP.merge_callbacks(callback_handlers...) -> MergedCallbacks

Several callback handlers as one: each is invoked in turn, and each reacts to the events it
handles. A handler is any value [`ReactiveMP.invoke_callback`](@ref) takes.

# Examples

```jldoctest
julia> struct PrintEvent <: ReactiveMP.Event{:print_event}
           label::String
       end;

julia> handler1 = (print_event = (event) -> println("first: ", event.label),);

julia> handler2 = (print_event = (event) -> println("second: ", event.label),);

julia> ReactiveMP.invoke_callback(ReactiveMP.merge_callbacks(handler1, handler2), PrintEvent("hello"));
first: hello
second: hello
```

See also [`ReactiveMP.handle_event`](@ref).
"""
function merge_callbacks(callback_handlers...)
    return MergedCallbacks(callback_handlers)
end

function invoke_callback(merged::MergedCallbacks, event::Event)
    for callback in merged.callbacks
        invoke_callback(callback, event)
    end
    return event
end

"""
    ReactiveMP.generate_span_id(callbacks)

The identifier shared by a "before" event and its "after" event, such as
[`ReactiveMP.BeforeMessageRuleCallEvent`](@ref) and [`ReactiveMP.AfterMessageRuleCallEvent`](@ref):
a `UUIDs.uuid4()`, or `nothing` when `callbacks` is `nothing`. A handler of a custom type may add a
method returning `nothing` to skip generating them; a [`ReactiveMP.MergedCallbacks`](@ref) always
generates them.
"""
function generate_span_id end

function generate_span_id(::Nothing)
    return nothing
end

function generate_span_id(callbacks)
    return uuid4()
end

# Internal helper used by the `Base.show` methods of the event types defined in
# this file. Honors the `:compact` `IOContext` flag and the `nothing` span:
#
# - emits nothing at all when `span_id === nothing` (no `span=nothing` noise),
# - otherwise emits a leading `, ` separator followed by either
#     * `span=<first 4 hex chars>…` when `get(io, :compact, false) == true`
#       (greppable: the same prefix appears on the matching `Before`/`After`
#       pair), or
#     * `span_id=<full uuid>` for non-compact contexts (REPL, Pluto, …) where
#       the full identifier is preferable.
function _show_span(io::IO, span_id)
    if isnothing(span_id)
        return nothing
    end
    if get(io, :compact, false)
        s = string(span_id)
        print(io, ", span=", first(s, 4), "…")
    else
        print(io, ", span_id=", span_id)
    end
    return nothing
end

# All defined events go here, so its easier to document them all in one place

"""
    ReactiveMP.BeforeMessageRuleCallEvent{M, Ms, Mr, S} <: Event{:before_message_rule_call}

The event right before a message rule runs, at each call of a [`ReactiveMP.MessageMapping`](@ref).

# Fields

- `mapping`: the [`ReactiveMP.MessageMapping`](@ref), which holds the node, the target and the
  algorithm;
- `messages`: the inbound messages the rule reads, a tuple, or `nothing` for none;
- `marginals`: the marginals the rule reads, a tuple, or `nothing` for none;
- `span_id`: the identifier shared with the [`ReactiveMP.AfterMessageRuleCallEvent`](@ref) (see
  [`ReactiveMP.generate_span_id`](@ref)).
"""
struct BeforeMessageRuleCallEvent{M, Ms, Mr, S} <:
    Event{:before_message_rule_call}
    mapping::M
    messages::Ms
    marginals::Mr
    span_id::S
end

"""
    ReactiveMP.AfterMessageRuleCallEvent{M, Ms, Mr, R, A, L, S} <: Event{:after_message_rule_call}

The event right after a message rule ran, or its fallback, or after a `missing` input gave a
`missing` message.

# Fields

- `mapping`: the [`ReactiveMP.MessageMapping`](@ref), which holds the node, the target and the
  algorithm;
- `messages`: the inbound messages the rule read, a tuple, or `nothing` for none;
- `marginals`: the marginals the rule read, a tuple, or `nothing` for none;
- `result`: the message's data, what the rule or the fallback returned, or `missing`;
- `annotations`: the message's [`ReactiveMP.AnnotationDict`](@ref);
- `logscale`: the message's log scale (see [`getlogscale`](@ref)), or `nothing` where log scales
  are not tracked;
- `span_id`: the identifier shared with the [`ReactiveMP.BeforeMessageRuleCallEvent`](@ref).
"""
struct AfterMessageRuleCallEvent{M, Ms, Mr, R, A, L, S} <:
    Event{:after_message_rule_call}
    mapping::M
    messages::Ms
    marginals::Mr
    result::R
    annotations::A
    logscale::L
    span_id::S
end

"""
    ReactiveMP.BeforeProductOfTwoMessagesEvent{V, C, L, R, S} <: Event{:before_product_of_two_messages}

The event right before [`ReactiveMP.compute_product_of_two_messages`](@ref) multiplies two
messages.

# Fields

- `variable`: the [`AbstractVariable`](@ref) the product is for;
- `context`: the [`ReactiveMP.MessageProductContext`](@ref);
- `left`, `right`: the two [`Message`](@ref)s;
- `span_id`: the identifier shared with the [`ReactiveMP.AfterProductOfTwoMessagesEvent`](@ref).
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
    ReactiveMP.AfterProductOfTwoMessagesEvent{V, C, L, R, Rs, A, S} <: Event{:after_product_of_two_messages}

The event right after [`ReactiveMP.compute_product_of_two_messages`](@ref) multiplied two
messages.

# Fields

- `variable`: the [`AbstractVariable`](@ref) the product is for;
- `context`: the [`ReactiveMP.MessageProductContext`](@ref);
- `left`, `right`: the two [`Message`](@ref)s;
- `result`: the product, a [`Message`](@ref);
- `annotations`: the product's [`ReactiveMP.AnnotationDict`](@ref);
- `span_id`: the identifier shared with the [`ReactiveMP.BeforeProductOfTwoMessagesEvent`](@ref).
"""
struct AfterProductOfTwoMessagesEvent{V, C, L, R, Rs, A, S} <:
    Event{:after_product_of_two_messages}
    variable::V
    context::C
    left::L
    right::R
    result::Rs
    annotations::A
    span_id::S
end

"""
    ReactiveMP.BeforeProductOfMessagesEvent{V, C, Ms, S} <: Event{:before_product_of_messages}

The event at the start of [`ReactiveMP.compute_product_of_messages`](@ref), before a collection of
messages is multiplied.

# Fields

- `variable`: the [`AbstractVariable`](@ref) the product is for;
- `context`: the [`ReactiveMP.MessageProductContext`](@ref);
- `messages`: the messages to multiply;
- `span_id`: the identifier shared with the [`ReactiveMP.AfterProductOfMessagesEvent`](@ref).
"""
struct BeforeProductOfMessagesEvent{V, C, Ms, S} <:
    Event{:before_product_of_messages}
    variable::V
    context::C
    messages::Ms
    span_id::S
end

"""
    ReactiveMP.AfterProductOfMessagesEvent{V, C, Ms, R, S} <: Event{:after_product_of_messages}

The event at the end of [`ReactiveMP.compute_product_of_messages`](@ref), after the fold and the
form constraint.

# Fields

- `variable`: the [`AbstractVariable`](@ref) the product is for;
- `context`: the [`ReactiveMP.MessageProductContext`](@ref);
- `messages`: the messages that were multiplied;
- `result`: the product, a [`Message`](@ref);
- `span_id`: the identifier shared with the [`ReactiveMP.BeforeProductOfMessagesEvent`](@ref).
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
    ReactiveMP.BeforeFormConstraintAppliedEvent{V, C, S, D, I} <: Event{:before_form_constraint_applied}

The event right before a product's form constraint applies, with [`constrain_form`](@ref), under
either strategy, [`FormConstraintCheckEach`](@ref) or [`FormConstraintCheckLast`](@ref).

# Fields

- `variable`: the [`AbstractVariable`](@ref) the product is for;
- `context`: the [`ReactiveMP.MessageProductContext`](@ref);
- `strategy`: the check strategy, `FormConstraintCheckEach()` or `FormConstraintCheckLast()`;
- `distribution`: the distribution about to be constrained;
- `span_id`: the identifier shared with the [`ReactiveMP.AfterFormConstraintAppliedEvent`](@ref).
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
    ReactiveMP.AfterFormConstraintAppliedEvent{V, C, S, D, R, I} <: Event{:after_form_constraint_applied}

The event right after a product's form constraint applied, with [`constrain_form`](@ref), under
either strategy, [`FormConstraintCheckEach`](@ref) or [`FormConstraintCheckLast`](@ref).

# Fields

- `variable`: the [`AbstractVariable`](@ref) the product is for;
- `context`: the [`ReactiveMP.MessageProductContext`](@ref);
- `strategy`: the check strategy, `FormConstraintCheckEach()` or `FormConstraintCheckLast()`;
- `distribution`: the distribution before the constraint;
- `result`: the distribution after it;
- `span_id`: the identifier shared with the [`ReactiveMP.BeforeFormConstraintAppliedEvent`](@ref).
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
    ReactiveMP.BeforeMarginalComputationEvent{V, C, Ms, S} <: Event{:before_marginal_computation}

The event right before a [`RandomVariable`](@ref) computes its marginal from its inbound
messages. Its callbacks are those of the variable's
`prod_context_for_marginal_computation` (see [`RandomVariableActivationOptions`](@ref)).

# Fields

- `variable`: the [`RandomVariable`](@ref);
- `context`: the [`ReactiveMP.MessageProductContext`](@ref) of the marginal;
- `messages`: the inbound messages;
- `span_id`: the identifier shared with the [`ReactiveMP.AfterMarginalComputationEvent`](@ref).
"""
struct BeforeMarginalComputationEvent{V, C, Ms, S} <:
    Event{:before_marginal_computation}
    variable::V
    context::C
    messages::Ms
    span_id::S
end

"""
    ReactiveMP.AfterMarginalComputationEvent{V, C, Ms, R, S} <: Event{:after_marginal_computation}

The event right after a [`RandomVariable`](@ref) computed its marginal from its inbound
messages. Its callbacks are those of the variable's `prod_context_for_marginal_computation`.

# Fields

- `variable`: the [`RandomVariable`](@ref);
- `context`: the [`ReactiveMP.MessageProductContext`](@ref) of the marginal;
- `messages`: the inbound messages;
- `result`: the marginal, a [`Marginal`](@ref);
- `span_id`: the identifier shared with the [`ReactiveMP.BeforeMarginalComputationEvent`](@ref).
"""
struct AfterMarginalComputationEvent{V, C, Ms, R, S} <:
    Event{:after_marginal_computation}
    variable::V
    context::C
    messages::Ms
    result::R
    span_id::S
end

# -----------------------------------------------------------------------------
# `Base.show` methods for the event types defined above.
#
# These honor the `:compact` `IOContext` flag (Julia stdlib convention, see
# `Base.show_circular` and friends in `base/show.jl`):
#
#   * `:compact => true`  — short, single-line form intended for trace
#     loggers like RxInfer's TensorBoardLoggerExt. Tuples of messages and
#     marginals are summarised by `nmsgs=N` / `nmarginals=N` and the span id
#     is truncated to a 4-char prefix that matches across `Before`/`After`
#     pairs.
#   * `:compact => false` (the default for REPL, Pluto, Jupyter)  — full
#     form: actual `messages` / `marginals` are printed, the span id is
#     emitted in full.
#
# Field order matches the struct definition so the output mirrors the
# event's own data. Variable identity is shown via `var=<label>` (every
# variable subtype stores a `label` field, see `src/variables/*.jl`).
# -----------------------------------------------------------------------------

# Best-effort label extraction. Variables (`RandomVariable`, `ConstVariable`,
# `DataVariable`) all carry a `label` field; fall back to the value itself
# for any other shape so the show methods stay total.
_var_label(v) = hasproperty(v, :label) ? getfield(v, :label) : v

# Count for `messages::Union{Tuple, Nothing}` and the marginals counterpart.
_count_or_zero(::Nothing) = 0
_count_or_zero(x) = length(x)

# Compact summary `nmsgs=N` vs. full `messages=<value>`. The compact form is
# what trace loggers want; the full form is what an interactive user wants
# when they `display(ev)` in a notebook.
function _show_messages_field(io::IO, name::String, value)
    if get(io, :compact, false)
        print(io, ", n", name, "=", _count_or_zero(value))
    else
        print(io, ", ", name, "=")
        show(io, value)
    end
    return nothing
end

function Base.show(io::IO, ev::BeforeMessageRuleCallEvent)
    print(io, "BeforeMessageRuleCallEvent(mapping=")
    show(io, ev.mapping)
    _show_messages_field(io, "msgs", ev.messages)
    _show_messages_field(io, "marginals", ev.marginals)
    _show_span(io, ev.span_id)
    print(io, ")")
    return nothing
end

function Base.show(io::IO, ev::AfterMessageRuleCallEvent)
    print(io, "AfterMessageRuleCallEvent(mapping=")
    show(io, ev.mapping)
    _show_messages_field(io, "msgs", ev.messages)
    _show_messages_field(io, "marginals", ev.marginals)
    print(io, ", result=")
    show(io, ev.result)
    print(io, ", annotations=")
    show(io, ev.annotations)
    ev.logscale === nothing || (print(io, ", logscale="); show(io, ev.logscale))
    _show_span(io, ev.span_id)
    print(io, ")")
    return nothing
end

function Base.show(io::IO, ev::BeforeProductOfTwoMessagesEvent)
    print(io, "BeforeProductOfTwoMessagesEvent(var=")
    show(io, _var_label(ev.variable))
    print(io, ", left=")
    show(io, ev.left)
    print(io, ", right=")
    show(io, ev.right)
    _show_span(io, ev.span_id)
    print(io, ")")
    return nothing
end

function Base.show(io::IO, ev::AfterProductOfTwoMessagesEvent)
    print(io, "AfterProductOfTwoMessagesEvent(var=")
    show(io, _var_label(ev.variable))
    print(io, ", left=")
    show(io, ev.left)
    print(io, ", right=")
    show(io, ev.right)
    print(io, ", result=")
    show(io, ev.result)
    print(io, ", annotations=")
    show(io, ev.annotations)
    _show_span(io, ev.span_id)
    print(io, ")")
    return nothing
end

function Base.show(io::IO, ev::BeforeProductOfMessagesEvent)
    print(io, "BeforeProductOfMessagesEvent(var=")
    show(io, _var_label(ev.variable))
    _show_messages_field(io, "messages", ev.messages)
    _show_span(io, ev.span_id)
    print(io, ")")
    return nothing
end

function Base.show(io::IO, ev::AfterProductOfMessagesEvent)
    print(io, "AfterProductOfMessagesEvent(var=")
    show(io, _var_label(ev.variable))
    _show_messages_field(io, "messages", ev.messages)
    print(io, ", result=")
    show(io, ev.result)
    _show_span(io, ev.span_id)
    print(io, ")")
    return nothing
end

function Base.show(io::IO, ev::BeforeFormConstraintAppliedEvent)
    print(io, "BeforeFormConstraintAppliedEvent(var=")
    show(io, _var_label(ev.variable))
    print(io, ", strategy=")
    show(io, ev.strategy)
    print(io, ", dist=")
    show(io, ev.distribution)
    _show_span(io, ev.span_id)
    print(io, ")")
    return nothing
end

function Base.show(io::IO, ev::AfterFormConstraintAppliedEvent)
    print(io, "AfterFormConstraintAppliedEvent(var=")
    show(io, _var_label(ev.variable))
    print(io, ", strategy=")
    show(io, ev.strategy)
    print(io, ", dist=")
    show(io, ev.distribution)
    print(io, ", result=")
    show(io, ev.result)
    _show_span(io, ev.span_id)
    print(io, ")")
    return nothing
end

function Base.show(io::IO, ev::BeforeMarginalComputationEvent)
    print(io, "BeforeMarginalComputationEvent(var=")
    show(io, _var_label(ev.variable))
    _show_messages_field(io, "messages", ev.messages)
    _show_span(io, ev.span_id)
    print(io, ")")
    return nothing
end

function Base.show(io::IO, ev::AfterMarginalComputationEvent)
    print(io, "AfterMarginalComputationEvent(var=")
    show(io, _var_label(ev.variable))
    _show_messages_field(io, "messages", ev.messages)
    print(io, ", result=")
    show(io, ev.result)
    _show_span(io, ev.span_id)
    print(io, ")")
    return nothing
end
