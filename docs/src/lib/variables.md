# [Variables](@id lib-variables)

A variable is an edge of the [factor graph](@ref concepts-factor-graphs): a quantity that the
[factor nodes](@extref MessagePassingRulesBase glossary-factor-node) connected to it share. The
engine has three kinds of variable, all subtypes of [`ReactiveMP.AbstractVariable`](@ref).

```@setup variables
using ReactiveMP, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, Rocket
import ReactiveMP: activate!, FactorNodeActivationOptions, get_stream_of_marginals,
    get_stream_of_predictions, set_initial_marginal!
include(joinpath(pkgdir(ReactiveMP), "docs", "nodes.jl"))
```

The examples use the `Gaussian` node of [The example node](@ref example-node).

## [Choosing the right variable type](@id lib-variables-choosing)

| Kind | Constructor | Holds | Changes by |
|------|-------------|-------|------------|
| [`RandomVariable`](@ref) | [`randomvar`](@ref) | a latent quantity, whose posterior you infer | inference, which updates its marginal |
| [`DataVariable`](@ref) | [`datavar`](@ref) | an observed quantity | [`new_observation!`](@ref) |
| [`ConstVariable`](@ref) | [`constvar`](@ref) | a fixed value | nothing: it is set at creation |

Use a random variable for every quantity you want a posterior for. Use a data variable for an
observation, especially one that changes between runs, as in streaming inference. Use a constant
for a hyperparameter or any other value that never changes.

```@example variables
x = randomvar(label = :x)
y = datavar(label = :y)
noise = constvar(1.0, label = :noise)
ReactiveMP.israndom(x), ReactiveMP.isdata(y), ReactiveMP.isconst(noise)
```

The `label` names the variable in displays, callback events and error messages.

## [Variables as reactive streams](@id lib-variables-streams)

A variable holds no single value. It holds reactive streams, which emit values once the graph is
live:

- a **marginal stream**, a [`ReactiveMP.MarginalObservable`](@ref), which emits the variable's
  [marginal](@extref MessagePassingRulesBase glossary-marginal) whenever it changes;
- one **message stream** per connected node, a [`ReactiveMP.MessageObservable`](@ref), which
  carries the [messages](@extref MessagePassingRulesBase glossary-message) between the variable
  and that node.

Connecting a node to a variable creates its message streams:

```@example variables
prior = factornode(Gaussian, [(:out, x), (:μ, constvar(0.0)), (:v, constvar(10.0))])
likelihood = factornode(Gaussian, [(:out, y), (:μ, x), (:v, noise)])
ReactiveMP.degree(x), ReactiveMP.degree(y), ReactiveMP.degree(noise)
```

The [`ReactiveMP.degree`](@ref) of a variable is its number of connections: `x` connects to both
nodes. The streams stay lazy while the graph is
[built](@ref concepts-inference-lifecycle-construction). [`ReactiveMP.activate!`](@ref) wires them
into a live network, variables first, then nodes:

```@example variables
activate!(x, RandomVariableActivationOptions())
activate!(y, DataVariableActivationOptions())
activate!(prior, FactorNodeActivationOptions())
activate!(likelihood, FactorNodeActivationOptions())

posteriors = Marginal[]
subscription = subscribe!(get_stream_of_marginals(x), (q) -> push!(posteriors, q))
new_observation!(y, 2.0)
last(posteriors)
```

The observation propagates through the graph: the likelihood sends a message to `x`, and `x`
multiplies it with the prior's message. [Inference lifecycle](@ref concepts-inference-lifecycle)
describes the order of construction, activation and observation.

```@docs
ReactiveMP.AbstractVariable
ReactiveMP.activate!
```

## [Common variable API](@id lib-variables-common)

Every kind of variable answers the same questions: which kind it is, how many nodes it connects
to, and what its streams are.

### Type predicates

```@docs
ReactiveMP.israndom
ReactiveMP.isdata
ReactiveMP.isconst
ReactiveMP.degree
```

### Marginal and message streams

[`ReactiveMP.get_stream_of_marginals`](@ref) returns a variable's marginal stream, and activation
connects it with [`ReactiveMP.set_stream_of_marginals!`](@ref). A late subscriber receives the
latest marginal at once.

Some graphs need a value before inference computes one. In a
[variational](@extref MessagePassingRulesBase glossary-vmp) graph, a rule may read a marginal that
depends, through other rules, on the rule's own message. You break the cycle with an initial
marginal, [`ReactiveMP.set_initial_marginal!`](@ref), or an
[initial message](@extref MessagePassingRulesBase glossary-initial-message),
[`ReactiveMP.set_initial_message!`](@ref). The chain below has a
[mean-field](@extref MessagePassingRulesBase glossary-mean-field) node between `x1` and `x2`:

```@example variables
x1, x2, y2 = randomvar(label = :x1), randomvar(label = :x2), datavar(label = :y2)
chain = (
    factornode(Gaussian, [(:out, x1), (:μ, constvar(0.0)), (:v, constvar(10.0))]),
    factornode(Gaussian, [(:out, x2), (:μ, x1), (:v, constvar(1.0))], ((:out,), (:μ,), (:v,))),
    factornode(Gaussian, [(:out, y2), (:μ, x2), (:v, constvar(1.0))]),
)
foreach(v -> activate!(v, RandomVariableActivationOptions()), (x1, x2))
activate!(y2, DataVariableActivationOptions())
foreach(n -> activate!(n, FactorNodeActivationOptions()), chain)
set_initial_marginal!(x1, NormalMeanVariance(0.0, 1.0))

q1 = Marginal[]
chain_subscription = subscribe!(get_stream_of_marginals(x1), (q) -> push!(q1, q))
for iteration in 1:20
    new_observation!(y2, 2.0)
end
mean(last(q1)), 2.0 * 10 / 12
```

Each observation runs one iteration. The mean of `q(x1)` converges to the exact posterior mean,
``2 \cdot 10 / 12``. Without the initial marginal, the middle node's rule towards `out` waits for
`q(x1)`. That marginal waits for the node's message towards `μ`, which waits for `q(x2)`, which
waits for the rule towards `out`. Nothing is computed.

```@docs
ReactiveMP.get_stream_of_marginals
ReactiveMP.set_stream_of_marginals!
ReactiveMP.set_initial_marginal!
ReactiveMP.set_initial_message!
```

### Prediction streams

A data variable's **prediction** is the product of the messages its nodes send to it: what the
model predicts for the observation without the observation itself. You ask for it when you
activate the variable, with the first field of [`DataVariableActivationOptions`](@ref):

```@example variables
xp, yp = randomvar(label = :x), datavar(label = :y)
nodes = (
    factornode(Gaussian, [(:out, xp), (:μ, constvar(0.0)), (:v, constvar(10.0))]),
    factornode(Gaussian, [(:out, yp), (:μ, xp), (:v, constvar(1.0))]),
)
activate!(xp, RandomVariableActivationOptions())
activate!(yp, DataVariableActivationOptions(true, false, nothing, nothing))
foreach(n -> activate!(n, FactorNodeActivationOptions()), nodes)

predictions = Marginal[]
prediction_subscription = subscribe!(get_stream_of_predictions(yp), (p) -> push!(predictions, p))
new_observation!(yp, 2.0)
last(predictions)
```

The prediction is the prior predictive, ``\mathcal{N}(0, 10 + 1)``. A random variable's
prediction stream is its marginal stream.

```@docs
ReactiveMP.get_stream_of_predictions
ReactiveMP.set_stream_of_predictions!
```

## [Random variables](@id lib-variables-random)

A random variable is a latent quantity. Its marginal is the product of all the messages its nodes
send to it, and the message it sends to one node is the product of the messages from all the
others. With more than one connection, it computes those products along an equality chain, which
shares the partial products between them (see [Internals](@ref internals-equality)).

```@docs
ReactiveMP.RandomVariable
ReactiveMP.randomvar
```

Its activation options say how it multiplies messages: two
[`ReactiveMP.MessageProductContext`](@ref)s, one for its outbound messages and one for its
marginal, and a [stream postprocessor](@ref lib-stream-postprocessors). The product contexts
carry the [form constraints](@ref custom-functional-form) and the product
[callbacks](@ref lib-callbacks).

```@docs
ReactiveMP.RandomVariableActivationOptions
ReactiveMP.activate!(::RandomVariable, ::RandomVariableActivationOptions)
```

## [Data variables](@id lib-variables-data)

A data variable is an observed quantity. It has no value until you give it one with
[`new_observation!`](@ref), and each new value propagates through the graph. The observation
reaches the nodes as a [point mass](@extref MessagePassingRulesBase glossary-point-mass), a
`PointMass` from BayesBase.

The value `missing` says that an observation is not available. A rule that depends on a
`missing` message gives `missing` in turn, and a product ignores a `missing` side. Here the
posterior of `x` falls back to its prior:

```@example variables
new_observation!(y, missing)
last(posteriors)
```

An observation must be a real number, an array of real numbers or a `UniformScaling`. Anything
else is an error that names the variable:

```@example variables
try
    new_observation!(y, Normal(0.0, 1.0))
catch err
    showerror(stdout, err)
end
```

```@docs
ReactiveMP.DataVariable
ReactiveMP.datavar
ReactiveMP.new_observation!
```

### [Non-standard observations](@id lib-variables-data-nonstandard)

A node of your own may take observations that are not numbers, such as text. You pass such a
value wrapped in a `PointMass` yourself, which [`new_observation!`](@ref) sends unchecked. The
node's rules dispatch on the concrete `PointMass` type and read the payload with
`BayesBase.getpointmass`.

The node below observes a text through its number of words, a Poisson count with rate `θ`. As a
function of the rate, the likelihood of `n` words is ``\theta^n e^{-\theta}``, a Gamma density
with shape ``n + 1`` and rate 1:

```@example variables
struct WordCount end

@define_factor_node(node = WordCount, type = Stochastic, interfaces = [:out, :θ])

@define_message_update_rule(
    node = WordCount, target = :θ,
    args = (m[:out]::PointMass{<:AbstractString},),
    body = (args) -> GammaShapeRate(length(split(BayesBase.getpointmass(args.m[:out]))) + 1, 1.0),
)

θ, text = randomvar(label = :θ), datavar(label = :text)
counter = factornode(WordCount, [(:out, text), (:θ, θ)])
activate!(θ, RandomVariableActivationOptions())
activate!(text, DataVariableActivationOptions())
activate!(counter, FactorNodeActivationOptions())

rates = Marginal[]
rate_subscription = subscribe!(get_stream_of_marginals(θ), (q) -> push!(rates, q))
new_observation!(text, PointMass("the quick brown fox"))
last(rates)
```

A `PointMass` of a non-numeric payload has limits you must respect:

- It has no `variate_form`, and therefore no `mean`, `var` or `logpdf`. A rule reads it with
  `BayesBase.getpointmass`, which is not exported. Calling `mean` on it recurses until the
  stack overflows.
- Only nodes written for the payload can connect to the variable. The numeric nodes of the rule
  packages compute with the moments of their inputs.
- The [Bethe free energy](@ref lib-score) is not defined for such an edge, since it needs
  log-densities.

`new_observation!` does not validate a `PointMass`: `PointMass(Beta(1, 1))` goes through as
readily as `PointMass("text")`. You make sure that only rules which understand the payload ever
receive it. [`@define_message_update_rule`](@extref MessagePassingRulesBase.@define_message_update_rule)
describes how to write them.

### Activation

A data variable's activation options ask for its prediction stream, or link its observations to
other variables through a function.

```@docs
ReactiveMP.DataVariableActivationOptions
ReactiveMP.activate!(::DataVariable, ::DataVariableActivationOptions)
```

## [Constant variables](@id lib-variables-constant)

A constant holds a fixed value as a clamped `PointMass`, whose
[log scale](@extref MessagePassingRulesBase glossary-log-scale) is zero. Its streams are complete
when you create it, so it needs no activation:

```@example variables
constant_subscription = subscribe!(get_stream_of_marginals(noise), (q) -> println(q))
nothing # hide
```

A constant receives no messages, and its marginal cannot be rewired.

```@docs
ReactiveMP.ConstVariable
ReactiveMP.constvar
```

## [How the streams are built](@id lib-variables-internals)

This section is for contributors who implement a variable or read the engine's code.

A node's interface calls [`ReactiveMP.create_new_stream_of_inbound_messages!`](@ref) once per
connection. The stream it returns carries the node's messages to the variable: the interface's
outbound message is the variable's inbound one.

- A **random variable** allocates a new inbound stream per connection. Activation creates as many
  outbound streams, fed by the equality chain, and connects the marginal stream to the product of
  the inbound messages. With one connection, its outbound message never emits.
- A **data variable** also allocates an inbound stream per connection; their product is its
  prediction. Every connected node reads the same outbound stream, of its observations.
- A **constant** counts its connections and returns the same stream, of its one message, to each
  of them. It has no inbound streams: asking for one, or setting its marginal or prediction
  stream, is an error.

```@docs
ReactiveMP.create_new_stream_of_inbound_messages!
```
