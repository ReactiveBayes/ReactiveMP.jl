# Callbacks in the Message Passing Procedure

ReactiveMP provides a way to "hook" into the message passing procedure and listen to various events
via "callbacks". This can be useful, for example, to debug messages or monitor the order of computations.

```@docs 
ReactiveMP.invoke_callback
```

## All defined events

Here is the list of predefined event types, to which a custom callback handler can react to.

```@docs 
ReactiveMP.BeforeMessageRuleCallback
ReactiveMP.AfterMessageRuleCallback
```
