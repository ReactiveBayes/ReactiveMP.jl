# Callbacks in the Message Passing Procedure

ReactiveMP provides a way to "hook" into the message passing procedure and listen to various
events via "callbacks". This can be useful, for example, to debug messages or monitor the
order of computations.

Each event fires with a `Val{:event_name}()` identifier and a structured **data** object
(a plain `Base.@kwdef struct`) that holds all relevant information as named fields.
Callback handlers receive both the event identifier and the data, and can access fields by
name (e.g. `data.variable`, `data.result`) instead of indexing into positional argument
tuples.

```@docs
ReactiveMP.invoke_callback
ReactiveMP.merge_callbacks
ReactiveMP.MergedCallbacks
```

## All defined events

Here is the list of predefined event identifiers and their corresponding data types.
A `NamedTuple` or `Dict{Symbol}` callback handler is keyed by the event symbol; a custom
handler struct implements `invoke_callback(::MyHandler, ::Val{:event_name}, data)`.

```@docs
ReactiveMP.BeforeMessageRuleCallback
ReactiveMP.AfterMessageRuleCallback
ReactiveMP.BeforeProductOfTwoMessages
ReactiveMP.AfterProductOfTwoMessages
ReactiveMP.BeforeProductOfMessages
ReactiveMP.AfterProductOfMessages
ReactiveMP.BeforeFormConstraintApplied
ReactiveMP.AfterFormConstraintApplied
ReactiveMP.BeforeMarginalComputation
ReactiveMP.AfterMarginalComputation
```

## Event data types

```@docs
ReactiveMP.BeforeMessageRuleCallbackData
ReactiveMP.AfterMessageRuleCallbackData
ReactiveMP.BeforeProductOfTwoMessagesData
ReactiveMP.AfterProductOfTwoMessagesData
ReactiveMP.BeforeProductOfMessagesData
ReactiveMP.AfterProductOfMessagesData
ReactiveMP.BeforeFormConstraintAppliedData
ReactiveMP.AfterFormConstraintAppliedData
ReactiveMP.BeforeMarginalComputationData
ReactiveMP.AfterMarginalComputationData
```
