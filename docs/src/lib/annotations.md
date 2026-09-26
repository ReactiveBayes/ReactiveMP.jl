# [Annotations](@id lib-annotations)

A message or a marginal holds a distribution. Annotations are an optional side channel that
travels with a message: values keyed by `Symbol`, such as a record of the inputs a message was
computed from, or debugging information. They are free when unused: nothing is allocated until the
first write. A message's log scale is not an annotation, but part of the message (see
[Log scales](@ref lib-logscale)).

## [The annotation dictionary](@id lib-annotations-dict)

Every message and marginal holds a [`ReactiveMP.AnnotationDict`](@ref), read with
[`getannotations`](@ref). The engine's functions read and write it; a rule does the same through
the base package's, [`getannotation`](@extref MessagePassingRulesBase.getannotation),
[`hasannotation`](@extref MessagePassingRulesBase.hasannotation) and
[`annotate!`](@extref MessagePassingRulesBase.annotate!), which the engine implements for it.

```@docs
ReactiveMP.AnnotationDict
ReactiveMP.annotate!(::ReactiveMP.AnnotationDict, ::Symbol, ::Any)
ReactiveMP.get_annotation
ReactiveMP.has_annotation
```

## [Annotation processors](@id lib-annotations-processors)

An annotation processor, a subtype of [`ReactiveMP.AbstractAnnotations`](@ref), decides what is
written, at three points:

- **before a rule runs**, [`ReactiveMP.pre_rule_annotations!`](@ref), with the new message's
  `AnnotationDict`, the [`ReactiveMP.MessageMapping`](@ref), and the rule's inbound messages and
  marginals: for annotations that do not depend on what the rule computes;
- **after a rule ran**, [`ReactiveMP.post_rule_annotations!`](@ref), with the same and the result:
  for annotations that do;
- **at a product of messages**, [`ReactiveMP.post_product_annotations!`](@ref), with the product's
  empty `AnnotationDict`, the two sides' annotations and the distributions: to merge the two into
  the product's.

A processor is given to both places it acts in: to the factor nodes, as the activation option
`annotations` of [`ReactiveMP.FactorNodeActivationOptions`](@ref), and to the variables, as the
`annotations` of their [`ReactiveMP.MessageProductContext`](@ref)s. Without the second, a product's
annotations are empty.

```@docs
ReactiveMP.AbstractAnnotations
ReactiveMP.pre_rule_annotations!
ReactiveMP.post_rule_annotations!
ReactiveMP.post_product_annotations!
```

## [A custom processor](@id lib-annotations-custom)

This processor counts the rule calls a message was computed from:

```julia
import ReactiveMP: AbstractAnnotations, AnnotationDict, has_annotation, get_annotation, annotate!

struct CountAnnotations <: AbstractAnnotations end

ReactiveMP.pre_rule_annotations!(::CountAnnotations, ann::AnnotationDict, mapping, messages, marginals) = nothing

function ReactiveMP.post_rule_annotations!(::CountAnnotations, ann::AnnotationDict, mapping, messages, marginals, result)
    annotate!(ann, :count, 1)
    return nothing
end

function ReactiveMP.post_product_annotations!(::CountAnnotations, merged::AnnotationDict, left_ann::AnnotationDict, right_ann::AnnotationDict, new_dist, left_dist, right_dist)
    count(ann) = has_annotation(ann, :count) ? get_annotation(ann, Int, :count) : 0
    annotate!(merged, :count, count(left_ann) + count(right_ann))
    return nothing
end

processors = (CountAnnotations(),)
FactorNodeActivationOptions(; annotations = processors)
MessageProductContext(; annotations = processors)
```

## [Input arguments](@id lib-annotations-input-arguments)

[`InputArgumentsAnnotations`](@ref) records what went into each rule call, the
[`ReactiveMP.MessageMapping`](@ref) (the node, the target and the algorithm), the inbound messages,
the marginals, and the result, and carries the record through the products of messages: the
provenance of a message travels with it. After a rule runs, a [`RuleInputArgumentsRecord`](@ref)
is stored under the key `:rule_input_arguments`; a product merges the two sides' records into a
[`ProductInputArgumentsRecord`](@ref), a flat list, however deeply the products nest.

```julia
record = get_rule_input_arguments(getannotations(message))

if record isa RuleInputArgumentsRecord
    println("rule: ", record.mapping, ", messages: ", record.messages, ", result: ", record.result)
elseif record isa ProductInputArgumentsRecord
    foreach(r -> println("contributed: ", r.mapping), record.mappings)
end
```

```@docs
InputArgumentsAnnotations
RuleInputArgumentsRecord
ProductInputArgumentsRecord
get_rule_input_arguments
```
