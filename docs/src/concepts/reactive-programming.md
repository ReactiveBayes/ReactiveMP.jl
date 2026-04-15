# [Reactive Programming Model](@id concepts-reactive-programming)

ReactiveMP.jl is built on a **reactive programming** paradigm. Unlike traditional inference engines that follow a pre-defined, static computation schedule (e.g., performing a forward and backward pass), ReactiveMP.jl operates by reacting to changes in the underlying data.

## The Mental Model

To use this package effectively, it helps to shift your thinking from "static values" to "dynamic streams."

### 1. Observables as Streams
In many algorithms, a message is just a piece of data at a specific point in time. In ReactiveMP.jl, messages and marginals are treated as **Observables**. 

Think of an Observable not as a single number, but as a **stream** of values. Whenever a node performs a computation and produces a new result, it "emits" this value into the stream. Any downstream node listening to this stream will automatically receive the update.

### 2. Automatic Propagation
Because nodes are connected via these streams, the graph handles its own execution. You do not need to manually trigger a "message passing step." Instead:
- An external event occurs (e.g., a new observation is added via [`new_observation!`](@ref)).
- This change triggers an update in a specific node.
- That node's output changes, which automatically notifies all connected neighbors.
- The change propagates through the graph structure, only visiting nodes that are actually affected by the update.

This "dependency-driven" execution ensures that we only perform the minimum amount of computation necessary to keep the beliefs up to date.

## For Deep Dives

The underlying machinery for this reactive behavior is provided by [Rocket.jl](https://github.com/ReactiveBayes/Rocket.jl). If you want to understand the low-level mechanics of how Observables, triggers, and reactive streams are implemented in Julia, we highly recommend exploring the Rocket.jl documentation.
