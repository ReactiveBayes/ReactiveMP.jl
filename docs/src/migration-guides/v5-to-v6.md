# Migration guide: v5 to v6

This guide covers the breaking changes introduced in ReactiveMP.jl v6 and how to update your code.

## Overview

v6 introduces two major changes:

1. **Annotations system** — the addon system is replaced by a new annotations system. Messages and marginals now carry an [`ReactiveMP.AnnotationDict`](@ref) instead of a typed tuple of addons. Annotation processors ([`ReactiveMP.AbstractAnnotations`](@ref) subtypes) handle post-processing externally.

2. **Renamed API** — many internal and public functions have been renamed to be more descriptive and consistent. The old names are removed; see the tables below for the mapping.

## Type parameter changes

`Message` and `Marginal` each lost one type parameter:

```julia
# v5
Message{D, A}
Marginal{D, A}

# v6
Message{D}
Marginal{D}
```

Code that dispatches on the second type parameter (e.g. `::Message{D, Nothing}`) must be updated to use only one parameter.

## Constructor changes

The fourth positional argument (`addons`) is replaced by an optional [`ReactiveMP.AnnotationDict`](@ref). In most cases you can simply drop the fourth argument:

```julia
# v5
Message(dist, false, false, nothing)
Marginal(dist, false, false, nothing)

# v6
Message(dist, false, false)
Marginal(dist, false, false)
```

## Renamed functions

### Variable API

| v5 | v6 |
|---|---|
| `update!(datavar, value)` | `new_observation!(datavar, value)` |
| `getmarginal(variable)` | `ReactiveMP.get_stream_of_marginals(variable)` |
| `getmarginals(variables)` | `map(ReactiveMP.get_stream_of_marginals, variables)` |
| `getprediction(variable)` | `ReactiveMP.get_stream_of_predictions(variable)` |
| `getpredictions(variables)` | `map(ReactiveMP.get_stream_of_predictions, variables)` |
| `setmarginal!(variable, value)` | `ReactiveMP.set_initial_marginal!(variable, value)` |
| `setmarginals!(variables, values)` | `ReactiveMP.set_initial_marginal!.(variables, values)` |
| `setmessage!(variable, value)` | `ReactiveMP.set_initial_message!(variable, value)` |
| `setmessages!(variables, values)` | `ReactiveMP.set_initial_message!.(variables, values)` |

### Node interface API

| v5 | v6 |
|---|---|
| `messagein(interface)` | `ReactiveMP.get_stream_of_inbound_messages(interface)` |
| `messageout(interface)` | `ReactiveMP.get_stream_of_outbound_messages(interface)` |
| `create_messagein!(variable)` | `ReactiveMP.create_new_stream_of_inbound_messages!(variable)` |

### Skip strategy API

| v5 | v6 |
|---|---|
| `SkipInitial()` as strategy argument | `skip_initial()` as a pipe operator |
| `SkipClamped()` as strategy argument | `skip_clamped()` as a pipe operator |
| `SkipClampedAndInitial()` as strategy argument | `skip_clamped_and_initial()` as a pipe operator |
| `IncludeAll()` | *(no filter needed)* |

The old strategies were passed to `getmarginal` as a second argument. In v6 the observable returned by `ReactiveMP.get_stream_of_marginals` is filtered directly with a pipe:

```julia
# v5
obs = getmarginal(variable, SkipInitial())

# v6
obs = ReactiveMP.get_stream_of_marginals(variable) |> skip_initial()
```

### Annotation / addon functions

| v5 | v6 | Notes |
|---|---|---|
| `getaddons(msg)` | `getannotations(msg)` | Works on both `Message` and `Marginal` |
| `getlogscale(msg)` | `getlogscale(getannotations(msg))` | No longer a direct method on messages/marginals |
| `getmemory(msg)` | `get_rule_input_arguments(getannotations(msg))` | Renamed concept: "memory" is now "input arguments" |
| `getmemoryaddon(msg)` | *removed* | Use `get_rule_input_arguments(getannotations(msg))` |

## Removed types and functions

The following exports no longer exist in v6:

| Removed | Replacement |
|---|---|
| `AbstractAddon` | [`ReactiveMP.AbstractAnnotations`](@ref) |
| `AddonLogScale` | [`ReactiveMP.LogScaleAnnotations`](@ref) |
| `AddonMemory` | [`ReactiveMP.InputArgumentsAnnotations`](@ref) |
| `AddonDebug` | *removed* (use [callbacks](@ref lib-callbacks) instead) |
| `multiply_addons` | [`ReactiveMP.post_product_annotations!`](@ref) |
| `@invokeaddon` | *removed* (macros like `@logscale` call `annotate!` directly) |
| `message_mapping_addons` | *removed* |
| `message_mapping_addon` | *removed* |
| `MarginalSkipStrategy` | `skip_initial()`, `skip_clamped()`, `skip_clamped_and_initial()` filter operators |
| `SkipClamped` | `skip_clamped()` |
| `SkipInitial` | `skip_initial()` |
| `SkipClampedAndInitial` | `skip_clamped_and_initial()` |
| `IncludeAll` | *(no filter needed)* |
| `apply_skip_filter` | *removed* |
| `as_marginal_observable` | *removed* |
| `AbstractVariable` | `ReactiveMP.AbstractVariable` (no longer exported) |
| `update!` | `new_observation!` |
| `getmarginal`, `getmarginals` | `ReactiveMP.get_stream_of_marginals` |
| `getprediction`, `getpredictions` | `ReactiveMP.get_stream_of_predictions` |
| `setmarginal!`, `setmarginals!` | `ReactiveMP.set_initial_marginal!` |
| `setmessage!`, `setmessages!` | `ReactiveMP.set_initial_message!` |
| `messagein` | `ReactiveMP.get_stream_of_inbound_messages` |
| `messageout` | `ReactiveMP.get_stream_of_outbound_messages` |
| `create_messagein!` | `ReactiveMP.create_new_stream_of_inbound_messages!` |

## Writing custom rules

Rules continue to return only the distribution result — no change needed for most rules. The `@logscale` macro now writes directly into the [`ReactiveMP.AnnotationDict`](@ref) instead of going through `@invokeaddon`:

```julia
# v5
@rule MyNode(:out, Marginalisation) (m_in::PointMass,) = begin
    result = compute_something(m_in)
    @logscale 0
    return result
end

# v6 — identical, no changes needed
@rule MyNode(:out, Marginalisation) (m_in::PointMass,) = begin
    result = compute_something(m_in)
    @logscale 0
    return result
end
```

Inside a `@rule` body, `getannotations()` (previously `getaddons()`) returns the [`ReactiveMP.AnnotationDict`](@ref) for the current rule execution. The `@logscale value` macro is a shorthand for `annotate!(getannotations(), :logscale, value)`.

## Testing rules with `@call_rule`

The `@call_rule` macro no longer supports `return_addons` or `addons` keyword arguments. Use the `annotations` keyword to pass an [`ReactiveMP.AnnotationDict`](@ref) and read it back after the call:

```julia
# v5
result, addons = @call_rule [ return_addons = true ] MyNode(:out, Marginalisation) (
    m_in = PointMass(1.0), addons = (AddonLogScale(),)
)
logscale = getlogscale(addons)

# v6
ann = AnnotationDict()
result = @call_rule MyNode(:out, Marginalisation) (m_in = PointMass(1.0), annotations = ann,)
logscale = getlogscale(ann)
```

## Writing custom annotation processors

If you had a custom `AbstractAddon` subtype, migrate it to an [`ReactiveMP.AbstractAnnotations`](@ref) subtype. See the [Annotations overview](@ref lib-annotations) for a complete guide.

```julia
# v5
struct MyAddon <: AbstractAddon end

# Had to implement multiply_addons and handle tuple-based dispatch

# v6
struct MyAnnotations <: AbstractAnnotations end

# Implement these two methods:
function ReactiveMP.post_rule_annotations!(::MyAnnotations, ann::AnnotationDict, mapping, messages, marginals, result)
    annotate!(ann, :my_key, compute_something(result))
end

function ReactiveMP.post_product_annotations!(::MyAnnotations, merged::AnnotationDict, left_ann::AnnotationDict, right_ann::AnnotationDict, new_dist, left_dist, right_dist)
    # Merge annotations from left and right into merged
end
```

See [`ReactiveMP.post_rule_annotations!`](@ref) and [`ReactiveMP.post_product_annotations!`](@ref).

## Configuring annotations in RxInfer

When setting up inference with RxInfer, replace addon configuration with the equivalent annotation processors:

```julia
# v5
addons = (AddonLogScale(), AddonMemory())

# v6
annotations = (LogScaleAnnotations(), InputArgumentsAnnotations())
```

Refer to the RxInfer.jl documentation for the updated inference configuration API.
