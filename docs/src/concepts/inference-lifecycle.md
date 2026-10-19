# [Inference lifecycle](@id concepts-inference-lifecycle)

Every inference run goes through three phases: **construction**, **activation** and
**observation**. This page runs the three phases on one small graph, block by block.
[Getting started](@ref getting-started) runs the same phases and adds the free energy.

!!! note
    With [RxInfer.jl](https://reactivebayes.github.io/RxInfer.jl/stable/), its `infer` function
    manages these phases. This page is for working with the engine directly.

The graph is one factor, the `Gaussian` node of [the example node](@ref example-node), a normal
density with a known variance. It relates an observation `y` to a latent mean `x`.

```@example lifecycle
using ReactiveMP, MessagePassingRulesBase, BayesBase, ExponentialFamily, Rocket
import ReactiveMP: activate!, FactorNodeActivationOptions, get_stream_of_marginals
include(joinpath(pkgdir(ReactiveMP), "docs", "nodes.jl"))
nothing # hide
```

## [Phase 1: Construction](@id concepts-inference-lifecycle-construction)

You create the variables and factor nodes of the model, and connect them. Each variable is
created by its role (see [Variables](@ref lib-variables)): [`randomvar`](@ref) for a latent
quantity you infer, [`datavar`](@ref) for an observation, and [`constvar`](@ref) for a value that
never changes.

```@example lifecycle
x = randomvar(label = :x)
y = datavar(label = :y)
v = constvar(2.0)
nothing # hide
```

A [`factornode`](@ref) connects a node type to the variables on its interfaces. An optional third
argument gives the [factorisation](@extref MessagePassingRulesBase glossary-factorisation) of the
node's local marginals; without it, all interfaces form one cluster.

```@example lifecycle
node = factornode(Gaussian, [(:out, y), (:μ, x), (:v, v)])
```

Each connection allocates the variable's stream of messages for that edge, a
[`ReactiveMP.MessageObservable`](@ref). All streams are **lazy**: they exist, but compute nothing.

```
  [datavar: y] ──── [factor: node] ──── (randomvar: x)
                     unconnected         unconnected
                     streams             streams
```

!!! note
    A variable's connections are fixed during construction: a variable does not see a node
    created after the variable is activated.

## [Phase 2: Activation](@id concepts-inference-lifecycle-activation)

Activation wires the lazy streams into a live network. You call [`ReactiveMP.activate!`](@ref) on
each variable, then on each factor node, with an options object:

- A random variable takes [`RandomVariableActivationOptions`](@ref): the
  [`ReactiveMP.MessageProductContext`](@ref)s its outbound messages and its marginal are computed
  with, and a [stream postprocessor](@ref lib-stream-postprocessors).
- A data variable takes [`DataVariableActivationOptions`](@ref): whether to compute its
  predictions, and whether its values are a function of other variables.
- A factor node takes [`ReactiveMP.FactorNodeActivationOptions`](@ref): the algorithm its rules
  run under, callbacks, annotations, diagnostics, services for its rules, a rule fallback, and
  whether to track log scales (see [Activation options](@ref lib-activation-options)).
- A constant needs no activation.

```@example lifecycle
activate!(x, RandomVariableActivationOptions())
activate!(y, DataVariableActivationOptions())
activate!(node, FactorNodeActivationOptions())
```

Variational message passing and loopy graphs need initial marginals or messages to start. You set
them between the two steps, with [`ReactiveMP.set_initial_marginal!`](@ref) and
[`ReactiveMP.set_initial_message!`](@ref). This graph is a tree under belief propagation and needs
none.

```
  [datavar: y] ──── [factor: node] ──── (randomvar: x) ──► marginal q(x)
       ▲                rules                streams
  (waiting for          connected            connected
   observations)
```

After activation, every edge carries a stream subscribed to its sources. The marginal of `x`, a
[`ReactiveMP.MarginalObservable`](@ref), emits a new belief whenever the messages it is formed
from change. Nothing is computed until something subscribes: to a marginal, with
[`ReactiveMP.get_stream_of_marginals`](@ref), or to the free energy, with
[`bethe_free_energy`](@ref).

```@example lifecycle
subscription = subscribe!(get_stream_of_marginals(x), (q) -> println("q(x) updated: ", getdata(q)))
nothing # hide
```

## [Phase 3: Observation](@id concepts-inference-lifecycle-observation)

Observations drive inference. You give them to the data variables with
[`new_observation!`](@ref):

```@example lifecycle
new_observation!(y, 3.14)
```

The observation is a `PointMass(3.14)` message on the data variable's outbound stream. It
propagates through the connected node, whose rule computes its message towards `x`, and on to
`x`, whose marginal updates. Without a prior, the marginal of `x` is that message,
``\mathcal{N}(x \mid 3.14, 2)``.

```
  new_observation!(y, 3.14)
         │
         ▼
  [datavar: y] ──► message ──► [factor: node] ──► message ──► (randomvar: x)
                                                                     │
                                                                     ▼
                                                              marginal q(x) emits
```

Every call to [`new_observation!`](@ref) propagates again. In streaming inference, each call is a
new data point. In variational inference, the same data is given once per iteration, and each
iteration updates the marginals and the free energy.

```@example lifecycle
new_observation!(y, 1.0)
```

A value that `PointMass` can represent is observed this way: a real number, an array of real
numbers or a `UniformScaling`. Anything else is an error. Data that is deliberately not numeric,
read by a custom node, is wrapped explicitly (see
[Non-standard observations](@ref lib-variables-data-nonstandard)).

When the run is over, unsubscribe:

```@example lifecycle
unsubscribe!(subscription)
```

## [Summary](@id concepts-inference-lifecycle-summary)

| Phase | What happens | Key functions |
|-------|-------------|---------------|
| **Construction** | Variables and nodes are created and connected; the streams are allocated, lazy | [`randomvar`](@ref), [`datavar`](@ref), [`constvar`](@ref), [`factornode`](@ref) |
| **Activation** | The streams are wired into a live network; initial values are set; subscriptions are made | [`ReactiveMP.activate!`](@ref), [`ReactiveMP.FactorNodeActivationOptions`](@ref), [`ReactiveMP.get_stream_of_marginals`](@ref) |
| **Observation** | Data arrives, messages propagate, marginals update | [`new_observation!`](@ref) |

## [Next steps](@id concepts-inference-lifecycle-next)

- [Factor nodes](@ref lib-node): how nodes are created and activated.
- [Variables](@ref lib-variables): the streams of each kind of variable.
- [Callbacks](@ref lib-callbacks): observing every rule call and product.
- [Form constraints](@ref custom-functional-form): constraining the form of marginals.
