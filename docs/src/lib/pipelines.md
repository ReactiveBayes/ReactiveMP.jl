# [Pipelines](@id lib-pipelines)

A **pipeline** is a composable sequence of transformations applied to the message stream on a specific edge of a factor node. Pipelines are attached during the [Activation](@ref lib-node-activation) phase via [`ReactiveMP.FactorNodeActivationOptions`](@ref) and run every time a new message is emitted on that edge.

Pipelines are useful for:
- **Debugging** — logging every message that flows through a particular edge.
- **Scheduling** — redirecting computation to a specific thread or scheduler.
- **Early termination** — stopping propagation on an edge under custom conditions.
- **Asynchrony** — decoupling computation from the emission of the upstream message.

## [Available pipeline stages](@id lib-pipelines-stages)

| Stage | Purpose |
|-------|---------|
| [`ReactiveMP.EmptyPipelineStage`](@ref) | No-op; the default when no pipeline is attached |
| [`ReactiveMP.LoggerPipelineStage`](@ref) | Prints each message to stdout as it passes through |
| [`ReactiveMP.DiscontinuePipelineStage`](@ref) | Stops propagation — drops the message and emits nothing |
| [`ReactiveMP.AsyncPipelineStage`](@ref) | Schedules the downstream computation asynchronously |
| [`ReactiveMP.ScheduleOnPipelineStage`](@ref) | Redirects computation to a custom Rocket.jl scheduler |
| [`ReactiveMP.CompositePipelineStage`](@ref) | Chains two stages together in sequence |

## [Composing pipeline stages](@id lib-pipelines-compose)

Stages are composed by wrapping them in a [`ReactiveMP.CompositePipelineStage`](@ref). The `|>` operator is the idiomatic way to chain stages:

```julia
pipeline = LoggerPipelineStage() |> ScheduleOnPipelineStage(my_scheduler)
```

This applies the logger first, then redirects to `my_scheduler`.

## [Attaching a pipeline to a node](@id lib-pipelines-attach)

Pipelines are provided when activating a factor node via [`ReactiveMP.FactorNodeActivationOptions`](@ref). In practice this is done through the model specification layer (e.g. RxInfer.jl's `@model` macro with `pipeline` metadata), but at the low level it looks like:

```julia
options = ReactiveMP.FactorNodeActivationOptions(
    factorization,
    metadata,
    pipeline,        # <-- pipeline stage for this node's edges
    dependencies,
    scheduler,
)
ReactiveMP.activate!(node, options)
```

!!! note
    A pipeline stage is applied to *all* edges of the node it is attached to. To apply different stages to different edges, use [`ReactiveMP.FactorNodeActivationOptions`](@ref) with per-interface configuration.

## [Custom pipeline stages](@id lib-pipelines-custom)

Custom stages are created by subtyping [`ReactiveMP.AbstractPipelineStage`](@ref) and implementing [`ReactiveMP.apply_pipeline_stage`](@ref):

```julia
struct MyStage <: ReactiveMP.AbstractPipelineStage end

function ReactiveMP.apply_pipeline_stage(stage::MyStage, factornode, tag, stream)
    # transform `stream` (a Rocket.jl observable) and return the modified stream
    return stream |> map(eltype(stream), msg -> (println("Intercepted: ", msg); msg))
end
```

```@docs
ReactiveMP.AbstractPipelineStage
ReactiveMP.apply_pipeline_stage
ReactiveMP.EmptyPipelineStage
ReactiveMP.CompositePipelineStage
ReactiveMP.LoggerPipelineStage
ReactiveMP.DiscontinuePipelineStage
ReactiveMP.AsyncPipelineStage
ReactiveMP.ScheduleOnPipelineStage
ReactiveMP.schedule_updates
```
