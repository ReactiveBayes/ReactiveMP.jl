# [Inference lifecycle](@id concepts-inference-lifecycle)

Every inference computation in ReactiveMP.jl goes through three phases: **construction**, **activation**, and **observation**. Understanding these phases is essential when working directly with the engine.

!!! note
    If you are using ReactiveMP.jl through [RxInfer.jl](https://github.com/reactivebayes/RxInfer.jl), these phases are managed for you automatically by the `infer` function. This page is aimed at users working with the low-level API directly.

## [Phase 1: Construction](@id concepts-inference-lifecycle-construction)

In the construction phase, you create the variables and factor nodes of your model and connect them together.

**Variables** are created with one of three constructors depending on their role:

```julia
x = randomvar()   # latent variable — will be inferred
y = datavar()     # observed quantity — will receive data
c = constvar(2.0) # fixed constant — never changes
```

See [Variables](@ref lib-variables) for a full description of each type.

**Factor nodes** are connected to variables using the `make_node` machinery (typically called by a model specification layer). Each connection registers the variable with the node and allocates a [`ReactiveMP.MessageObservable`](@ref) stream for that edge. At this point, all streams are **lazy** — they exist as placeholders but are not yet computing anything.

After construction, the graph looks like this conceptually:

```
  [datavar: y] ──── [factor: f] ──── (randomvar: x)
                     unconnected         unconnected
                     streams             streams
```

!!! note
    The degree of a variable (number of connected factors) is determined during construction. Adding connections after activation is not supported.

## [Phase 2: Activation](@id concepts-inference-lifecycle-activation)

Activation wires the lazy observable streams into a live reactive network. This is done by calling [`ReactiveMP.activate!`](@ref) on each variable and factor node, passing an options object that bundles inference-time configuration.

For factor nodes, activation is driven by [`ReactiveMP.FactorNodeActivationOptions`](@ref), which carries:
- The factorization assumption (mean-field, structured, or full BP).
- An optional [stream postprocessor](@ref lib-stream-postprocessors) applied to outbound message, marginal, and score streams (e.g. for scheduling).
- Metadata and approximation method settings.

For variables, activation is driven by [`ReactiveMP.RandomVariableActivationOptions`](@ref) or [`ReactiveMP.DataVariableActivationOptions`](@ref), which wire up the marginal stream and prediction stream.

After activation, the graph is live:

```
  [datavar: y] ──── [factor: f] ──── (randomvar: x) ──► marginal q(x)
       ▲               rules                streams
  (waiting for         connected            connected
   observations)
```

Every edge now carries a [`ReactiveMP.MessageObservable`](@ref) that is subscribed to its upstream sources. The marginal at `x` is connected to a [`ReactiveMP.MarginalObservable`](@ref) that will emit updated beliefs every time a message changes.

## [Phase 3: Observation](@id concepts-inference-lifecycle-observation)

Once the graph is activated, inference is driven by feeding data into the data variables using [`new_observation!`](@ref):

```julia
new_observation!(y, 3.14)
```

This call pushes a new [`Message`](@ref) wrapping a `PointMass(3.14)` into the data variable's outbound stream. The change propagates reactively through all connected factor nodes, triggering rule computations, which in turn push updated messages to downstream variables, which update their marginals.

The result is that subscribing to the marginal stream of `x` yields updated posterior beliefs automatically:

```
  new_observation!(y, 3.14)
         │
         ▼
  [datavar: y] ──► message ──► [factor: f] ──► message ──► (randomvar: x)
                                                                  │
                                                                  ▼
                                                           marginal q(x) emits
```

You can subscribe to the marginal stream of any [`RandomVariable`](@ref) to receive updated beliefs:

```julia
subscribe!(get_stream_of_marginals(x), (marginal) -> println("Updated: ", mean(marginal)))
```

Multiple calls to [`new_observation!`](@ref) are possible after activation — each one triggers another round of reactive propagation. This makes the engine suitable for streaming/online inference scenarios.

## [Summary](@id concepts-inference-lifecycle-summary)

| Phase | What happens | Key functions |
|-------|-------------|---------------|
| **Construction** | Variables and nodes created, edges connected, streams allocated (lazy) | [`randomvar`](@ref), [`datavar`](@ref), [`constvar`](@ref), [`@node`](@ref) |
| **Activation** | Lazy streams wired into a live reactive network | [`ReactiveMP.activate!`](@ref), [`ReactiveMP.FactorNodeActivationOptions`](@ref) |
| **Observation** | Data fed in, messages propagate, marginals update | [`new_observation!`](@ref), [`ReactiveMP.get_stream_of_marginals`](@ref) |

## [Next steps](@id concepts-inference-lifecycle-next)

- [Factor nodes](@ref lib-node) — how nodes are implemented and activated.
- [Variables](@ref lib-variables) — stream creation and activation details for each variable type.
- [Callbacks](@ref lib-callbacks) — how to hook into message and marginal computation events.
- [Custom functional form](@ref custom-functional-form) — constraining the functional form of marginals during inference.
