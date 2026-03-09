# Events in the Message Passing Procedure

ReactiveMP provides a way to "hook" into the message passing procedure and listen to various events
via "event handlers". This can be useful, for example, to debug messages or monitor the order of computations.

```@docs 
ReactiveMP.Event
ReactiveMP.handle_event
ReactiveMP.broadcast_event
```

## All defined events

Here is the list of predefined events, to which a custom event handler can subscribe to.

```@docs 
ReactiveMP.BeforeMessageRuleCallEvent
ReactiveMP.AfterMessageRuleCallEvent
```
