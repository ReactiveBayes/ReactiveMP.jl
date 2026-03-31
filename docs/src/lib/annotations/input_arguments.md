# [Input arguments annotations](@id lib-annotations-input-arguments)

## Background: tracing rule inputs

During inference, every message flowing along an edge is computed by a message
update rule. `InputArgumentsAnnotations` records what went into each rule call —
the `MessageMapping` (which node and interface the rule was for), the incoming
messages, the incoming marginals, and the result distribution — and propagates
that record through subsequent message products.

This is useful for debugging and for implementing callbacks that need to inspect
the full provenance of a message: rather than re-running or re-examining the
model structure, the record travels with the message itself.

## What gets stored

After each rule execution a [`RuleInputArgumentsRecord`](@ref) is written into
the message's annotation dict under the `:rule_input_arguments` key.  When two
messages are multiplied, their records are merged into a
[`ProductInputArgumentsRecord`](@ref) that contains all contributing records as
a flat list, regardless of how deeply nested the products were.

## Reading input arguments from a message

```julia
using ReactiveMP

# ann is the AnnotationDict of some message
record = get_rule_input_arguments(ann)

if record isa RuleInputArgumentsRecord
    println("single rule: ", record.mapping)
    println("messages:  ", record.messages)
    println("marginals: ", record.marginals)
    println("result:    ", record.result)
elseif record isa ProductInputArgumentsRecord
    for r in record.mappings
        println("contributed rule: ", r.mapping)
    end
end
```

## API

```@docs
ReactiveMP.InputArgumentsAnnotations
ReactiveMP.RuleInputArgumentsRecord
ReactiveMP.ProductInputArgumentsRecord
ReactiveMP.get_rule_input_arguments
```
