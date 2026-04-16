# Migration guide: v5 to v6

This guide covers the breaking changes introduced in ReactiveMP.jl v6 and how to update your code.

## Overview

v6 introduces three major changes:

1. **Annotations system** — the addon system is replaced by a new annotations system. Messages and marginals now carry an [`ReactiveMP.AnnotationDict`](@ref) instead of a typed tuple of addons. Annotation processors ([`ReactiveMP.AbstractAnnotations`](@ref) subtypes) handle post-processing externally.

2. **Stream postprocessors** — the `AbstractPipelineStage` API and the per-node `scheduler` argument have been unified into a single [`ReactiveMP.AbstractStreamPostprocessor`](@ref) abstraction that postprocesses outbound message streams, marginal streams, and score streams uniformly.

3. **Renamed API** — many internal and public functions have been renamed to be more descriptive and consistent. The old names are removed; see the tables below for the mapping.

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

## [From `AbstractPipelineStage` + scheduler to `AbstractStreamPostprocessor`](@id v5-to-v6-stream-postprocessors)

The `AbstractPipelineStage` hierarchy and the separate node-level `scheduler` argument have been replaced by a single [`ReactiveMP.AbstractStreamPostprocessor`](@ref) abstraction. The new API is described in detail on the [Stream postprocessors](@ref lib-stream-postprocessors) page; this section summarises the mechanical migration.

### Activation options

`FactorNodeActivationOptions` lost both its `pipeline` and `scheduler` positional fields and gained a single `postprocessor` field:

```julia
# v5 / early v6
FactorNodeActivationOptions(metadata, dependencies, pipeline, annotations, scheduler, rulefallback, callbacks)

# v6
FactorNodeActivationOptions(metadata, dependencies, postprocessor, annotations, rulefallback, callbacks)
```

`RandomVariableActivationOptions` had its `scheduler` field renamed to `stream_postprocessor`:

```julia
# v5 / early v6
RandomVariableActivationOptions(AsapScheduler(), prod_context_msg, prod_context_marginal)

# v6
RandomVariableActivationOptions(nothing, prod_context_msg, prod_context_marginal)
# or with an explicit postprocessor:
RandomVariableActivationOptions(ScheduleOnStreamPostprocessor(PendingScheduler()), prod_context_msg, prod_context_marginal)
```

`nothing` is the no-op postprocessor: each `postprocess_stream_of_*` method has a `::Nothing` pass-through fallback.

### Type and helper renaming

| v5 | v6 |
|---|---|
| `AbstractPipelineStage` | [`ReactiveMP.AbstractStreamPostprocessor`](@ref) |
| `EmptyPipelineStage()` / `collect_pipeline(_, nothing)` | `nothing` (uses the `::Nothing` pass-through fallback) |
| `CompositePipelineStage(stages)` | [`ReactiveMP.CompositeStreamPostprocessor`](@ref)`(stages)` |
| `ScheduleOnPipelineStage(scheduler)` | [`ReactiveMP.ScheduleOnStreamPostprocessor`](@ref)`(scheduler)` |
| `apply_pipeline_stage(stage, factornode, tag, stream)` | [`ReactiveMP.postprocess_stream_of_outbound_messages`](@ref)`(postprocessor, stream)` |
| `getscheduler(options)` | `getpostprocessor(options)` |
| `getpipeline(options)` | `getpostprocessor(options)` |
| `collect_pipeline(_, ...)` | *removed* — postprocessors are passed through unchanged |
| `+` composition of stages | wrap in `CompositeStreamPostprocessor((left, right))` |

### Removed pipeline stages

The following pipeline stages are gone with no direct replacement:

| Removed | Replacement |
|---|---|
| `LoggerPipelineStage` | Use [callbacks](@ref lib-callbacks) (e.g. message-product / post-rule callbacks) instead — they observe the same events without subscribing to the streams. |
| `AsyncPipelineStage` | Wrap a Rocket.jl `AsyncScheduler` in a [`ReactiveMP.ScheduleOnStreamPostprocessor`](@ref). |
| `DiscontinuePipelineStage` | Removed; was unused. Implement a custom `AbstractStreamPostprocessor` if needed. |
| `schedule_updates(vars; pipeline_stage = ...)` | Construct a [`ReactiveMP.ScheduleOnStreamPostprocessor`](@ref) and pass it via [`ReactiveMP.RandomVariableActivationOptions`](@ref). |

### Custom pipeline stages

If you implemented a custom `AbstractPipelineStage`, port it to `AbstractStreamPostprocessor`. The stage signature loses the `factornode` and `tag` arguments — postprocessors operate on streams uniformly and have no node context:

```julia
# v5 / early v6
struct MyStage <: ReactiveMP.AbstractPipelineStage end
function ReactiveMP.apply_pipeline_stage(::MyStage, factornode, tag, stream)
    return stream |> ...
end

# v6
struct MyStreamPostprocessor <: ReactiveMP.AbstractStreamPostprocessor end
ReactiveMP.postprocess_stream_of_outbound_messages(::MyStreamPostprocessor, stream) = stream |> ...
ReactiveMP.postprocess_stream_of_marginals(::MyStreamPostprocessor, stream)         = stream
ReactiveMP.postprocess_stream_of_scores(::MyStreamPostprocessor, stream)            = stream
```

Implement only the stream kinds you actually attach the postprocessor to; if a stream of an unsupported kind reaches it during activation, a `MethodError` is raised. To pass a kind through unchanged, return the stream as-is.

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
| `AbstractPipelineStage` and subtypes (`EmptyPipelineStage`, `CompositePipelineStage`, `ScheduleOnPipelineStage`, `LoggerPipelineStage`, `AsyncPipelineStage`, `DiscontinuePipelineStage`), `apply_pipeline_stage`, `collect_pipeline`, `schedule_updates`, `getpipeline`, `getscheduler` | See the [stream postprocessor migration section](@ref v5-to-v6-stream-postprocessors) above |

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
