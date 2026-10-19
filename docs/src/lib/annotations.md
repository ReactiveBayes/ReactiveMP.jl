# [Annotations](@id lib-annotations)

A message or a marginal holds a distribution. Its **annotations** are optional metadata that
travel with it: values keyed by `Symbol`, such as a record of the inputs a message was computed
from. They cost nothing when unused. A message or marginal that nothing may annotate shares one
frozen, empty dictionary, and the engine gives a fresh one only to what may write to it: the
annotation processors, and a rule that takes the `ann` slot. A message's
[log scale](@ref lib-logscale) is not an annotation, but part of the message.

```@setup annotations
using ReactiveMP, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, Rocket
import ReactiveMP: activate!, FactorNodeActivationOptions, MessageProductContext, get_stream_of_marginals
include(joinpath(pkgdir(ReactiveMP), "docs", "nodes.jl"))
```

The examples use the `Gaussian` node of [The example node](@ref example-node).

## [The annotation dictionary](@id lib-annotations-dict)

Every message and marginal holds a [`ReactiveMP.AnnotationDict`](@ref), which
[`getannotations`](@ref) returns:

```@example annotations
ann = ReactiveMP.AnnotationDict()
ReactiveMP.annotate!(ann, :source, "sensor 1")
ReactiveMP.has_annotation(ann, :source), ReactiveMP.get_annotation(ann, String, :source)
```

The engine's functions read and write it. A rule does the same through the base package's
functions, [`getannotation`](@extref MessagePassingRulesBase.getannotation),
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
written. The engine calls it at three points:

- **before a rule runs**, [`ReactiveMP.pre_rule_annotations!`](@ref), with the new message's
  `AnnotationDict`, the [`ReactiveMP.MessageMapping`](@ref), and the rule's inbound messages and
  marginals. It writes the annotations that do not depend on the rule's result.
- **after a rule ran**, [`ReactiveMP.post_rule_annotations!`](@ref), with the same arguments and
  the result. It writes the annotations that do.
- **at a product of messages**, [`ReactiveMP.post_product_annotations!`](@ref), with the
  product's empty `AnnotationDict`, the two sides' annotations and the distributions. It merges
  the two sides' annotations into the product's.

You give a processor to both places it acts in: to the factor nodes, as the activation option
`annotations` of [`ReactiveMP.FactorNodeActivationOptions`](@ref), and to the variables, as the
`annotations` of their [`ReactiveMP.MessageProductContext`](@ref)s. Without the second, a
product's annotations are empty.

```@docs
ReactiveMP.AbstractAnnotations
ReactiveMP.pre_rule_annotations!
ReactiveMP.post_rule_annotations!
ReactiveMP.post_product_annotations!
```

## [A custom processor](@id lib-annotations-custom)

The processor below counts the rule calls a message was computed from. After a rule, the count
is one. A product adds the counts of its two sides:

```@example annotations
import ReactiveMP: AbstractAnnotations, AnnotationDict, has_annotation, get_annotation, annotate!

struct CountAnnotations <: AbstractAnnotations end

ReactiveMP.pre_rule_annotations!(::CountAnnotations, ann::AnnotationDict, mapping, messages, marginals) = nothing

function ReactiveMP.post_rule_annotations!(::CountAnnotations, ann::AnnotationDict, mapping, messages, marginals, result)
    annotate!(ann, :count, 1)
    return nothing
end

function ReactiveMP.post_product_annotations!(::CountAnnotations, merged::AnnotationDict, left::AnnotationDict, right::AnnotationDict, new_dist, left_dist, right_dist)
    count(ann) = has_annotation(ann, :count) ? get_annotation(ann, Int, :count) : 0
    annotate!(merged, :count, count(left) + count(right))
    return nothing
end
nothing # hide
```

The function below builds a latent `x` with a normal prior, observed through a second node, gives
the processors to every node and to `x`, and returns the posterior's annotations:

```@example annotations
function posterior_annotations(processors)
    x, y = randomvar(label = :x), datavar(label = :y)
    prior = factornode(Gaussian, [(:out, x), (:μ, constvar(0.0)), (:v, constvar(10.0))])
    likelihood = factornode(Gaussian, [(:out, y), (:μ, x), (:v, constvar(1.0))])
    context = MessageProductContext(; annotations = processors)
    activate!(x, RandomVariableActivationOptions(nothing, context, context))
    activate!(y, DataVariableActivationOptions())
    foreach(n -> activate!(n, FactorNodeActivationOptions(; annotations = processors)), (prior, likelihood))
    posteriors = Marginal[]
    subscription = subscribe!(get_stream_of_marginals(x), (q) -> push!(posteriors, q))
    new_observation!(y, 2.0)
    unsubscribe!(subscription)
    return getannotations(last(posteriors))
end

posterior_annotations((CountAnnotations(),))
```

The posterior is the product of two messages, each computed by one rule call.

## [Input arguments](@id lib-annotations-input-arguments)

[`InputArgumentsAnnotations`](@ref) records what went into each rule call: the
[`ReactiveMP.MessageMapping`](@ref) (the node, the target and the algorithm), the inbound
messages, the marginals and the result. It carries the record through the products of messages,
so the provenance of a message travels with it. After a rule runs, a
[`RuleInputArgumentsRecord`](@ref) is stored under the key `:rule_input_arguments`. A product
merges the two sides' records into a [`ProductInputArgumentsRecord`](@ref), a flat list, however
deeply the products nest. [`get_rule_input_arguments`](@ref) reads the record:

```@example annotations
record = get_rule_input_arguments(posterior_annotations((InputArgumentsAnnotations(),)))
```

A `ProductInputArgumentsRecord` lists its rule calls in `mappings`, each a
`RuleInputArgumentsRecord`:

```@example annotations
[(r.mapping.target, r.result) for r in record.mappings]
```

```@docs
InputArgumentsAnnotations
RuleInputArgumentsRecord
ProductInputArgumentsRecord
get_rule_input_arguments
```
