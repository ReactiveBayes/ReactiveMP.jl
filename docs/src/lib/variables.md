
# [Variables](@id lib-variables)

Variables are fundamental building blocks of a [factor graph](@ref concepts-factor-graphs). Each variable represents either a latent quantity to be inferred, an observed data point, or a fixed constant. All variable types are subtypes of [`ReactiveMP.AbstractVariable`](@ref).

## [Choosing the right variable type](@id lib-variables-choosing)

There are three kinds of variables, each with a distinct role:

| Type | Constructor | Role | Can be updated? |
|------|-------------|------|----------------|
| [`ReactiveMP.RandomVariable`](@ref) | [`ReactiveMP.randomvar`](@ref) | Latent quantity to be inferred | No — inference updates its marginal |
| [`ReactiveMP.DataVariable`](@ref) | [`ReactiveMP.datavar`](@ref) | Observed quantity that receives data | Yes — via [`new_observation!`](@ref) |
| [`ReactiveMP.ConstVariable`](@ref) | [`ReactiveMP.constvar`](@ref) | Fixed constant, never changes | No — wired at construction time |

The choice of variable type affects how the engine allocates streams and handles messages:

- Use `randomvar` for any quantity you want to infer a posterior over.
- Use `datavar` for observations that may change between inference calls (e.g., in online or streaming settings).
- Use `constvar` for fixed hyperparameters, known constants, or any value that will never change.

## [Variables as reactive streams](@id lib-variables-streams)

In ReactiveMP.jl, a variable is not a single value — it is a source of reactive *streams*. Each variable holds:

- A **marginal stream** ([`ReactiveMP.MarginalObservable`](@ref)) that emits updated [`Marginal`](@ref) beliefs as inference progresses.
- One **message stream** ([`ReactiveMP.MessageObservable`](@ref)) per connected factor node, carrying messages flowing between the variable and that node.

These streams are *lazy*: they are allocated during [construction](@ref concepts-inference-lifecycle-construction) but carry no values until the graph is [activated](@ref concepts-inference-lifecycle-activation). After activation, feeding new data into a `datavar` triggers automatic propagation through the network, updating marginals reactively.

See [Inference lifecycle](@ref concepts-inference-lifecycle) for an overview of the construction → activation → observation flow.

```@docs
ReactiveMP.AbstractVariable
```

## [Common variable API](@id lib-variables-common)

All variable types share a common interface for querying their kind, degree, and reactive streams.

### Type predicates

```@docs
ReactiveMP.israndom
ReactiveMP.isdata
ReactiveMP.isconst
ReactiveMP.degree
```

### Message stream allocation

```@docs
ReactiveMP.create_new_stream_of_inbound_messages!
```

### Marginal and message streams

Every variable exposes a *marginal stream* — a reactive observable that emits updated `Marginal` values as inference progresses. The stream is accessed via [`ReactiveMP.get_stream_of_marginals`](@ref) and wired up during graph activation via [`ReactiveMP.set_stream_of_marginals!`](@ref). Initial beliefs can be seeded before inference starts with [`ReactiveMP.set_initial_marginal!`](@ref), and initial messages with [`ReactiveMP.set_initial_message!`](@ref).

```@docs
ReactiveMP.get_stream_of_marginals
ReactiveMP.set_stream_of_marginals!
ReactiveMP.set_initial_marginal!
ReactiveMP.set_initial_message!
```

### Prediction streams

A *prediction stream* gives an estimate of what the variable's value would look like from the model's perspective — without conditioning on observed data for that variable. It is accessed via [`ReactiveMP.get_stream_of_predictions`](@ref) and connected during graph activation.

```@docs
ReactiveMP.get_stream_of_predictions
ReactiveMP.set_stream_of_predictions!
```

## [Random variables](@id lib-variables-random)

Random variables represent latent (unobserved) quantities in the model. During inference, messages flow through them to update the marginal belief.

```@docs
ReactiveMP.RandomVariable
ReactiveMP.randomvar
```

### Stream creation

A `RandomVariable` starts empty: its `input_messages` and `output_messages` collections are empty vectors. Each time a factor node connects to the variable, `ReactiveMP.create_new_stream_of_inbound_messages!` is called, which allocates a new `MessageObservable{AbstractMessage}`, appends it to `input_messages`, and returns it together with its index. The returned stream becomes the *outbound* message stream from the factor node's perspective (the message the node will send toward the variable). At this point, the degree equals the number of connected nodes. All streams are unconnected (lazy) until activation.

### Activation

```@docs
ReactiveMP.RandomVariableActivationOptions
ReactiveMP.activate!(::RandomVariable, ::RandomVariableActivationOptions)
```

The prediction stream for a `RandomVariable` is identical to its marginal stream, since there is no dedicated prediction channel for latent variables.

## [Data variables](@id lib-variables-data)

Data variables represent observed quantities. Their value is not fixed at creation time and can be updated later via [`new_observation!`](@ref).

```@docs
ReactiveMP.DataVariable
ReactiveMP.datavar
ReactiveMP.new_observation!
```

### Stream creation

A `DataVariable` has two distinct directions of information flow:

- **Outbound (observation) stream** — a `RecentSubject{Message}` stored in `messageout`. Calling [`new_observation!`](@ref) pushes a new `Message(PointMass(value), false, false)` into this subject. Every factor node connected to the variable receives the same shared `messageout` stream as its inbound message source; `ReactiveMP.get_stream_of_outbound_messages` always returns `messageout` regardless of the connection index.
- **Inbound (backward) messages** — each connecting factor node gets its own `MessageObservable{AbstractMessage}` allocated in `input_messages` via `ReactiveMP.create_new_stream_of_inbound_messages!`, the same way as for `RandomVariable`. These carry messages flowing *back* from the graph toward the data edge.

All streams are unconnected (lazy) until activation.

### Activation

```@docs
ReactiveMP.DataVariableActivationOptions
ReactiveMP.activate!(::DataVariable, ::DataVariableActivationOptions)
```

## [Constant variables](@id lib-variables-constant)

Constant variables hold a fixed value, wrapped in a `PointMass` distribution. Messages from constant variables are always marked as clamped.

```@docs
ReactiveMP.ConstVariable
ReactiveMP.constvar
```

### Stream creation

Unlike `RandomVariable` and `DataVariable`, a `ConstVariable` wires up its streams at *construction* time, not during graph activation. The constructor immediately connects:

- `messageout` to `of(Message(PointMass(constant), true, false))` — a single-element observable that emits one clamped message and completes.
- `marginal` to `of(Marginal(PointMass(constant), true, false))` — similarly fixed and clamped.

When a factor node connects to a `ConstVariable`, `ReactiveMP.create_new_stream_of_inbound_messages!` increments the `nconnected` counter (which defines [`ReactiveMP.degree`](@ref)) and returns the *same shared* `messageout` stream for every connection. There are no per-connection inbound streams: `ReactiveMP.get_stream_of_inbound_messages` raises an error because a `ConstVariable` never receives messages from nodes. Calling [`ReactiveMP.set_stream_of_marginals!`](@ref) or [`ReactiveMP.set_stream_of_predictions!`](@ref) also raises an error, since the streams are fixed and cannot be rewired. Constant variables require no activation step.
