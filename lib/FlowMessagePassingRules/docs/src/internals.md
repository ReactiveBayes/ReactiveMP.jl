# [Internals](@id flow-internals)

The helpers the rules are built on. None is public; a contributor changing the rules reads them.

The algorithm's accessors, and the two types the rules are declared under, one per method:

```@docs
FlowMessagePassingRules.getmodel
FlowMessagePassingRules.getmethod
FlowMessagePassingRules.FlowLinearization
FlowMessagePassingRules.FlowUnscented
```

The linearisation rules for a message in precision form share these, one per direction:

```@docs
FlowMessagePassingRules.flow_forward_precision
FlowMessagePassingRules.flow_backward_precision
```

The unscented rules push the sigma points through the model themselves:

```@docs
FlowMessagePassingRules.flow_unscented
FlowMessagePassingRules.flow_unscented_statistics
```
