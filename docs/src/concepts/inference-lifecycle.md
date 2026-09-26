# [Inference lifecycle](@id concepts-inference-lifecycle)

Every inference run goes through three phases: **construction**, **activation** and
**observation**. [Getting started](@ref getting-started) runs all three on a small model.

!!! note
    Through [RxInfer.jl](https://github.com/reactivebayes/RxInfer.jl), its `infer` function
    manages these phases. This page is for working with the engine directly.

## [Phase 1: Construction](@id concepts-inference-lifecycle-construction)

The variables and factor nodes of the model are created and connected.

**Variables** are created by their role (see [Variables](@ref lib-variables)):

```julia
x = randomvar()   # latent: inferred
y = datavar()     # observed: receives data
c = constvar(2.0) # constant: never changes
```

**Factor nodes** are created with [`factornode`](@ref), from the node type, the variables on each
interface, and the factorisation of the node's local marginals:

```julia
node = factornode(NormalMeanVariance, [(:out, y), (:μ, x), (:v, c)], ((:out, :μ), (:v,)))
```

Each connection allocates the variable's stream of messages for that edge, a
[`ReactiveMP.MessageObservable`](@ref). All streams are **lazy**: they exist, but compute nothing.

```
  [datavar: y] ──── [factor: f] ──── (randomvar: x)
                     unconnected         unconnected
                     streams             streams
```

!!! note
    A variable's connections are fixed during construction: a node created after the variable
    is activated is not seen by it.

## [Phase 2: Activation](@id concepts-inference-lifecycle-activation)

Activation wires the lazy streams into a live network: [`ReactiveMP.activate!`](@ref) on each
variable, then on each factor node, with an options object.

- A random variable takes [`RandomVariableActivationOptions`](@ref): the
  [`ReactiveMP.MessageProductContext`](@ref)s its outbound messages and its marginal are computed
  with, and a [stream postprocessor](@ref lib-stream-postprocessors).
- A data variable takes [`DataVariableActivationOptions`](@ref): whether to compute its
  predictions, and whether its values are a function of other variables.
- A factor node takes [`ReactiveMP.FactorNodeActivationOptions`](@ref): the algorithm its rules
  run under, callbacks, annotations, diagnostics, services for its rules, a rule fallback, and
  whether to track log scales (see [Activation options](@ref lib-activation-options)).
- A constant needs no activation.

Initial marginals and messages, which variational message passing and loopy graphs need to
start, are set between the two, with [`ReactiveMP.set_initial_marginal!`](@ref) and
[`ReactiveMP.set_initial_message!`](@ref).

```
  [datavar: y] ──── [factor: f] ──── (randomvar: x) ──► marginal q(x)
       ▲               rules                streams
  (waiting for         connected            connected
   observations)
```

Every edge now carries a stream subscribed to its sources, and the marginal of `x`, a
[`ReactiveMP.MarginalObservable`](@ref), emits a new belief whenever the messages it is formed from
change. Nothing is computed until something subscribes: to the marginals, with
[`ReactiveMP.get_stream_of_marginals`](@ref), and to the free energy, with
[`bethe_free_energy`](@ref).

```julia
subscribe!(get_stream_of_marginals(x), (marginal) -> println("Updated: ", mean(marginal)))
```

## [Phase 3: Observation](@id concepts-inference-lifecycle-observation)

Inference is driven by observations, given to the data variables with
[`new_observation!`](@ref):

```julia
new_observation!(y, 3.14)
```

The observation is a `PointMass(3.14)` message on the data variable's outbound stream. It
propagates through the connected nodes, whose rules compute their messages, and on to the
variables, whose marginals update.

```
  new_observation!(y, 3.14)
         │
         ▼
  [datavar: y] ──► message ──► [factor: f] ──► message ──► (randomvar: x)
                                                                  │
                                                                  ▼
                                                           marginal q(x) emits
```

A value `PointMass` can represent, a real number, an array of real numbers or a
`UniformScaling`, is observed this way; anything else is an error. Data that is deliberately not
numeric, read by a custom node, is wrapped explicitly: see
[Non-standard observations](@ref lib-variables-data-nonstandard).

Every call to [`new_observation!`](@ref) propagates again. In streaming inference, each is a new
data point; in variational inference, the same data is given once per iteration, and each
iteration updates the marginals and the free energy.

## [Summary](@id concepts-inference-lifecycle-summary)

| Phase | What happens | Key functions |
|-------|-------------|---------------|
| **Construction** | Variables and nodes created, edges connected, streams allocated, lazy | [`randomvar`](@ref), [`datavar`](@ref), [`constvar`](@ref), [`factornode`](@ref) |
| **Activation** | Streams wired into a live network; initial values set; subscriptions made | [`ReactiveMP.activate!`](@ref), [`ReactiveMP.FactorNodeActivationOptions`](@ref), [`ReactiveMP.get_stream_of_marginals`](@ref) |
| **Observation** | Data given, messages propagate, marginals update | [`new_observation!`](@ref) |

## [Next steps](@id concepts-inference-lifecycle-next)

- [Factor nodes](@ref lib-node): how nodes are created and activated.
- [Variables](@ref lib-variables): the streams of each kind of variable.
- [Callbacks](@ref lib-callbacks): observing every rule call and product.
- [Form constraints](@ref custom-functional-form): constraining the form of marginals.
