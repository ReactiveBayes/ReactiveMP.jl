# [Annotations](@id lib-annotations)

Messages and marginals in ReactiveMP carry a probability distribution as their primary content. Annotations are an optional side-channel that can travel alongside a message, holding arbitrary extra information keyed by `Symbol`. Typical uses include tracking log-scale factors (see [`LogScaleAnnotations`](@ref lib-annotations-logscale)), recording which messages were used to compute a result, or attaching debugging information.

Annotations are designed to be zero-cost when unused: the underlying dictionary is only allocated on the first write.

## AnnotationDict

Every message and marginal holds an [`ReactiveMP.AnnotationDict`](@ref). The basic operations are:

```@docs
ReactiveMP.AnnotationDict
ReactiveMP.annotate!
ReactiveMP.get_annotation
ReactiveMP.has_annotation
```

## Annotation processors

Annotation processors are subtypes of [`ReactiveMP.AbstractAnnotations`](@ref) that define *how* annotations are written and merged. There are two integration points:

- **After a rule executes** — [`ReactiveMP.post_rule_annotations!`](@ref) is called with the processor, the rule's `AnnotationDict`, the `MessageMapping`, the incoming messages and marginals, and the result distribution. Use this to write annotations that depend on what the rule computed.
- **During a message product** — [`ReactiveMP.post_product_annotations!`](@ref) is called with the processor, a fresh merged `AnnotationDict`, and the left and right annotation dicts together with the distributions involved. Use this to merge annotations from the two incoming messages into the product message.

```@docs
ReactiveMP.AbstractAnnotations
ReactiveMP.post_rule_annotations!
ReactiveMP.post_product_annotations!
```

## Implementing a custom annotation processor

To add a new kind of annotation, subtype `AbstractAnnotations` and implement the two callbacks:

```julia
using ReactiveMP

# not exported by default
import ReactiveMP: AbstractAnnotations, AnnotationDict, has_annotation, get_annotation, annotate!

struct CountAnnotations <: AbstractAnnotations end

# Called after each rule execution
function ReactiveMP.post_rule_annotations!(::CountAnnotations, ann::AnnotationDict, mapping, messages, marginals, result)
    prev = has_annotation(ann, :count) ? get_annotation(ann, Int, :count) : 0
    annotate!(ann, :count, prev + 1)
    return nothing
end

# Called when two messages are multiplied
function ReactiveMP.post_product_annotations!(::CountAnnotations, merged::AnnotationDict, left_ann::AnnotationDict, right_ann::AnnotationDict, new_dist, left_dist, right_dist)
    left_count  = has_annotation(left_ann,  :count) ? get_annotation(left_ann,  Int, :count) : 0
    right_count = has_annotation(right_ann, :count) ? get_annotation(right_ann, Int, :count) : 0
    annotate!(merged, :count, left_count + right_count)
    return nothing
end
```

Processors are passed to `FactorNodeActivationOptions` (for rule-time annotation) and [`ReactiveMP.MessageProductContext`](@ref) (for product-time merging) when building a model. Both sites must be configured — see the RxInfer documentation for how to set this up at the model level.

## Built-in annotation processors

```@contents
Pages = ["annotations/logscale.md", "annotations/input_arguments.md"]
Depth = 1
```
