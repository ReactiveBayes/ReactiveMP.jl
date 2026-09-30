# [Message passing](@id concepts-message-passing)

Message passing is how ReactiveMP.jl performs inference on a [factor graph](@ref concepts-factor-graphs).
The engine never forms the full joint distribution. Instead, factor nodes and variables exchange
small local summaries with their neighbours, the
[messages](@extref MessagePassingRulesBase glossary-message). The posterior beliefs, the
[marginals](@extref MessagePassingRulesBase glossary-marginal), are formed from these messages.

```@setup message-passing
using ReactiveMP, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, Rocket
import ReactiveMP: activate!, FactorNodeActivationOptions, get_stream_of_marginals
include(joinpath(pkgdir(ReactiveMP), "docs", "nodes.jl"))
```

## [Belief propagation](@id concepts-message-passing-bp)

[Belief propagation](@extref MessagePassingRulesBase glossary-belief-propagation), the
sum-product algorithm, computes *exact* marginals on tree-shaped graphs. A message from a factor
node `f` towards a variable `x` summarises what `f` knows about `x` from the rest of the graph:

```math
\mu_{f \to x}(x) = \int f(x, y, z) \; \mu_{y \to f}(y) \; \mu_{z \to f}(z) \; \mathrm{d}y \; \mathrm{d}z
```

The message from `x` back towards `f` is the product of the messages that arrive at `x` from
every *other* factor. The marginal `q(x)` is the product of all the messages that arrive at `x`,
normalised.

On a graph with cycles, the engine iterates the same procedure. This is loopy belief
propagation, which typically converges to a good approximation.

![message](../assets/img/bp-message.svg)
*A belief propagation message*

## [Variational message passing](@id concepts-message-passing-vmp)

[Variational message passing](@extref MessagePassingRulesBase glossary-vmp) performs approximate
inference. It minimises the [Bethe free energy](@extref MessagePassingRulesBase glossary-bethe-free-energy),
a variational objective, under a
[factorisation](@extref MessagePassingRulesBase glossary-factorisation) of the posterior into
[clusters](@extref MessagePassingRulesBase glossary-cluster). ReactiveMP.jl implements this
general form for three reasons:

1. It includes exact belief propagation, as the case with no factorisation constraints.
2. It handles non-conjugate and complex models, with
   [mean-field](@extref MessagePassingRulesBase glossary-mean-field) or
   [structured](@extref MessagePassingRulesBase glossary-structured-vmp) factorisations.
3. It has a local, message-level form that fits the reactive computation model.

Under the mean-field factorisation, `q(x, y, z) = q(x) q(y) q(z)`, the message from `f` towards
`x` is

```math
\mu_{f \to x}(x) = \exp \int q(y) \, q(z) \log f(x, y, z) \; \mathrm{d}y \; \mathrm{d}z
```

It uses the *marginals* `q(y)` and `q(z)`, not the messages `μ(y)` and `μ(z)`.

![message](../assets/img/vmp-message.svg)
*A variational message under the structured factorisation q(x, y)q(z)*

## [Which rule computes a message](@id concepts-message-passing-dispatch)

Each message is computed by an [update rule](@extref MessagePassingRulesBase glossary-rule), an
ordinary Julia function that a rule package defines for a node, a target interface and an
[algorithm](@extref MessagePassingRulesBase glossary-algorithm). When you activate a node, the
engine decides what each rule reads from the node's algorithm and factorisation. By default, a
rule reads the messages on the other interfaces of its target's cluster, and the marginals of the
other clusters. So one cluster over every interface gives belief propagation, and one interface
per cluster gives mean-field variational message passing. A deterministic node's rules read the
messages on every other interface.

At each update, the engine finds the rule with
[`find_message_rule`](@extref MessagePassingRulesBase.find_message_rule). It looks at the node
type, the target, the node's algorithm and the types of the inputs. The algorithm is
[`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm) for most nodes. Messages
and marginals of different types select different rules, and where no rule matches, the error
lists the near misses. An algorithm may also declare its own
[dependencies](@extref MessagePassingRulesBase glossary-dependencies) with
[`@define_dependencies`](@extref MessagePassingRulesBase.@define_dependencies), and you choose a
node's algorithm at activation (see [Activation options](@ref lib-activation-options)).

## [The same graph, both ways](@id concepts-message-passing-demo)

The graph of [Getting started](@ref getting-started) has a latent `x` with the prior
``\mathcal{N}(x \mid 0, 10)`` and an observation `y` with the likelihood
``\mathcal{N}(y \mid x, 1)``. Both factors are the `Gaussian` node of
[the example node](@ref example-node). The function below builds the graph with a factorisation
of your choice, runs it on the observation `y = 2`, and prints every rule the engine runs through
a [callback](@ref lib-callbacks):

```@example message-passing
function run_graph(factorisation)
    x, y = randomvar(label = :x), datavar(label = :y)
    constants = (constvar(0.0), constvar(10.0), constvar(1.0))
    prior = factornode(Gaussian, [(:out, x), (:μ, constants[1]), (:v, constants[2])], factorisation)
    likelihood = factornode(Gaussian, [(:out, y), (:μ, x), (:v, constants[3])], factorisation)

    io = IOContext(stdout, :compact => true, :module => @__MODULE__)
    callbacks = (after_message_rule_call = (event) -> println(io, event.mapping, " → ", event.result),)
    activate!(x, RandomVariableActivationOptions())
    activate!(y, DataVariableActivationOptions())
    for node in (prior, likelihood)
        activate!(node, FactorNodeActivationOptions(; callbacks))
    end

    posterior, energy = Ref{Any}(), Ref{Float64}()
    subscriptions = [
        subscribe!(get_stream_of_marginals(x), (q) -> posterior[] = q),
        subscribe!(bethe_free_energy(Float64, (prior, likelihood), (x, y, constants...)), (f) -> energy[] = f),
    ]
    new_observation!(y, 2.0)
    foreach(unsubscribe!, subscriptions)
    return getdata(posterior[]), energy[]
end
nothing # hide
```

Under belief propagation, each node is one cluster, `(out, μ, v)`:

```@example message-passing
run_graph(((:out, :μ, :v),))
```

Each printed line is a [`ReactiveMP.MessageMapping`](@ref): the node, the target, and the inputs
the rule reads. `msgs` lists messages. The prior's rule towards `out` reads the messages on `μ`
and `v`, and the likelihood's rule towards `μ` reads the messages on `out` and `v`.

Under mean field, each interface is a cluster of its own:

```@example message-passing
run_graph(((:out,), (:μ,), (:v,)))
```

The same targets read `marginals` here, so the engine runs the node's variational rules. The two
rules towards `μ` differ in their inputs, which
[`which_message_update_rule`](@extref MessagePassingRulesBase.which_message_update_rule) shows
without running them. With messages, it finds the belief propagation rule:

```@example message-passing
which_message_update_rule(Gaussian, :μ; m = (out = PointMass(2.0), v = PointMass(1.0)))
```

With marginals, it finds the variational rule:

```@example message-passing
which_message_update_rule(Gaussian, :μ; q = (out = PointMass(2.0), v = PointMass(1.0)))
```

Both runs give the same posterior, ``\mathcal{N}(x \mid 2/1.1, 1/1.1)`` in its natural form, and
the same free energy, ``-\log \mathcal{N}(2 \mid 0, 11)``. Mean field is exact here because every
other edge of each node is a constant or an observation: a point mass, under which the expectation
of the log-density and the integral agree. Where two latent variables share a node, the two
factorisations give different messages. Mean field then needs initial marginals
([`ReactiveMP.set_initial_marginal!`](@ref)) and several iterations.

For the theory in depth, see the
[PhD dissertation](https://pure.tue.nl/ws/portalfiles/portal/313860204/20231219_Bagaev_hf.pdf)
that ReactiveMP.jl is based on.

## [Messages as streams](@id concepts-message-passing-reactive)

The word *reactive* refers to how messages are scheduled. Many message passing libraries build an
explicit schedule, such as forward and backward passes, before inference starts. ReactiveMP.jl
builds none:

- Every connection between a variable and a node carries a stream of messages, a
  [`ReactiveMP.MessageObservable`](@ref). Every variable carries a stream of marginals, a
  [`ReactiveMP.MarginalObservable`](@ref). Each emits a new value whenever its inputs change.
- When data arrives, with [`new_observation!`](@ref), the change propagates through the graph,
  and only the rules that depend on it run.
- The order of the updates follows from the graph and the data, not from a plan. In variational
  message passing, the order in which a rule's inputs are declared is the update schedule.

A message is computed lazily. A node emits a [`DeferredMessage`](@ref), which is computed when a
variable first reads it, so a message nobody needs is never computed.

The streams are [Rocket.jl](https://github.com/ReactiveBayes/Rocket.jl) observables. Nothing runs
until something subscribes: a subscription to a marginal, or to the free energy, pulls the
computation through the graph. This is why you first build a graph and then *activate* it, and
why the same graph serves streaming data, each observation propagating as it arrives (see
[Inference lifecycle](@ref concepts-inference-lifecycle)).

## [Messages and marginals](@id concepts-message-passing-types)

The engine wraps the two in their own types:

- A [`Message`](@ref) travels along one edge, from a node towards a variable or back.
- A [`Marginal`](@ref) is a belief about a variable, formed from the product of the messages
  that arrive at it. It can also be a belief about a cluster of a node's variables, computed by
  the node's marginal rule.

Both hold a distribution and forward its statistics, such as `mean` and `var`. Both record
whether the value is clamped, computed from constants and observations only, and whether it is
initial, set before inference. Both carry their
[log scale](@extref MessagePassingRulesBase glossary-log-scale) when log scales are tracked (see
[Log scales](@ref lib-logscale)), and optional [annotations](@ref lib-annotations).

## [Next steps](@id concepts-message-passing-next)

- [Messages](@ref lib-message) and [Marginals](@ref lib-marginal): the types and their streams.
- [Inference lifecycle](@ref concepts-inference-lifecycle): construction, activation and
  observation.
- [`MessagePassingRulesBase`](@extref MessagePassingRulesBase MessagePassingRulesBase): how
  nodes, rules and algorithms are defined.
