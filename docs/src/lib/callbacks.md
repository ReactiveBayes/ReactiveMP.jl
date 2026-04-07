# Callbacks in the Message Passing Procedure

ReactiveMP provides a way to "hook" into the message passing procedure and listen to various events
via "callbacks". This can be useful, for example, to debug messages or monitor the order of computations.

```@docs
ReactiveMP.Event
ReactiveMP.event_name
ReactiveMP.handle_event
ReactiveMP.invoke_callback
ReactiveMP.merge_callbacks
ReactiveMP.MergedCallbacks
```

## Event naming convention

Every event in ReactiveMP is a concrete subtype of [`ReactiveMP.Event{E}`](@ref) where `E` is a `Symbol` identifying the event.
The naming convention is straightforward: for an event identified by the symbol `:event_name`, the corresponding struct is called `EventNameEvent`.
For example:

| Symbol | Struct |
|--------|--------|
| `:before_message_rule_call` | [`ReactiveMP.BeforeMessageRuleCallEvent`](@ref) |
| `:after_product_of_two_messages` | [`ReactiveMP.AfterProductOfTwoMessagesEvent`](@ref) |
| `:before_form_constraint_applied` | [`ReactiveMP.BeforeFormConstraintAppliedEvent`](@ref) |

Each event struct carries the relevant data as fields, so you can inspect what happened during inference.
You can use [`ReactiveMP.event_name`](@ref) to retrieve the symbol from any event type:

```@example callbacks
using ReactiveMP #hide
ReactiveMP.event_name(ReactiveMP.BeforeProductOfTwoMessagesEvent)
```

To see which fields an event carries, use the standard Julia introspection:

```julia
julia> ?ReactiveMP.BeforeProductOfTwoMessagesEvent
```

## Event spans

Certain events create a "span". For example all "before" and "after" events 
can be considered together. To track these relationships ReactiveMP uses the 
`span_id` field in such events and uses the [`ReactiveMP.generate_span_id`](@ref)
function to generate shared ids.

```@docs 
ReactiveMP.generate_span_id
```

Custom callbacks can overwrite the `ReactiveMP.generate_span_id` to return `nothing`
if necessary. Note, however, that [`ReactiveMP.MergedCallbacks`](@ref) would still 
use the default implementation.

## All defined events

Here is the list of predefined event types, to which a custom callback handler can react to.

```@docs
ReactiveMP.BeforeMessageRuleCallEvent
ReactiveMP.AfterMessageRuleCallEvent
ReactiveMP.BeforeProductOfTwoMessagesEvent
ReactiveMP.AfterProductOfTwoMessagesEvent
ReactiveMP.BeforeProductOfMessagesEvent
ReactiveMP.AfterProductOfMessagesEvent
ReactiveMP.BeforeFormConstraintAppliedEvent
ReactiveMP.AfterFormConstraintAppliedEvent
ReactiveMP.BeforeMarginalComputationEvent
ReactiveMP.AfterMarginalComputationEvent
```
