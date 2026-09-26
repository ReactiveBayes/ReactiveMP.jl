# [Message passing](@id concepts-message-passing)

Message passing is how ReactiveMP.jl performs inference on a [factor graph](@ref concepts-factor-graphs).
Instead of computing the full joint distribution, factor nodes and variables exchange small,
local summaries, **messages**, with their neighbours. The posterior beliefs, the **marginals**,
are formed from these messages.

## [Belief propagation](@id concepts-message-passing-bp)

**Belief propagation**, the sum-product algorithm, computes *exact* marginals on tree-shaped
graphs. A message from a factor node `f` towards a variable `x` summarises what `f` knows about
`x` from the rest of the graph:

```math
\mu_{f \to x}(x) = \int f(x, y, z) \; \mu_{y \to f}(y) \; \mu_{z \to f}(z) \; \mathrm{d}y \; \mathrm{d}z
```

The message from `x` back towards `f` is the product of the messages arriving at `x` from every
*other* factor. The marginal `q(x)` is the product of all the messages arriving at `x`,
normalised.

On graphs with cycles, the same procedure is iterated (loopy belief propagation), and typically
converges to a good approximation.

![message](../assets/img/bp-message.svg)
*A belief propagation message*

## [Variational message passing](@id concepts-message-passing-vmp)

**Variational message passing** performs approximate inference by minimising the Bethe free
energy, a variational objective, under a **factorisation** of the posterior into clusters.
ReactiveMP.jl implements this general form because:

1. it includes exact belief propagation as the case of no factorisation constraints;
2. it handles non-conjugate and complex models with **mean-field** or **structured**
   factorisations;
3. it has a local, message-level form that fits the reactive computation model.

Under a mean-field factorisation, `q(x, y, z) = q(x) q(y) q(z)`, the message from `f` towards `x`
is

```math
\mu_{f \to x}(x) = \exp \int q(y) \, q(z) \log f(x, y, z) \; \mathrm{d}y \; \mathrm{d}z
```

It uses the *marginals* `q(y)` and `q(z)` rather than the messages `μ(y)` and `μ(z)`.

![message](../assets/img/vmp-message.svg)
*A variational message under the structured factorisation q(x, y)q(z)*

## [Which rule computes a message](@id concepts-message-passing-dispatch)

Each message is computed by an **update rule**, an ordinary Julia function a rule package
defines for a node, a target interface and an **algorithm**. When a node is activated, the engine
decides what each rule reads from the node's algorithm and factorisation: by default, the
messages on the other interfaces of the target's cluster, and the marginals of the other
clusters. A factorisation of one cluster over every interface therefore gives belief propagation,
and a factorisation of one interface per cluster mean-field variational message passing. A
deterministic node's rules read the messages on every other interface.

At each update, the engine finds the rule with
[`find_message_rule`](@extref MessagePassingRulesBase.find_message_rule), from the node type, the
target, the node's algorithm, [`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm)
for most nodes, and the types of the inputs: messages and marginals of different types select
different rules. Where no rule matches, the error lists the near misses. An algorithm may also
declare its own dependencies
([`@define_dependencies`](@extref MessagePassingRulesBase.@define_dependencies)), and a node's
user chooses its algorithm at activation (see [Activation options](@ref lib-activation-options)).

For the theory in depth, see the
[PhD dissertation](https://pure.tue.nl/ws/portalfiles/portal/313860204/20231219_Bagaev_hf.pdf)
ReactiveMP.jl is based on.

## [Messages as streams](@id concepts-message-passing-reactive)

The word *reactive* refers to how messages are scheduled. Many message passing libraries build an
explicit schedule, such as forward and backward passes, before inference starts. ReactiveMP.jl
builds none:

- every connection between a variable and a node carries a stream of messages, a
  [`ReactiveMP.MessageObservable`](@ref), and every variable a stream of marginals, a
  [`ReactiveMP.MarginalObservable`](@ref), which emit a new value whenever their inputs change;
- when data arrives, with [`new_observation!`](@ref), the change propagates through the graph,
  and only the rules that depend on it run;
- the order of the updates follows from the graph and the data, not from a plan; in variational
  message passing, the order in which a rule's inputs are declared is the update schedule.

A message is computed lazily: a node emits a [`DeferredMessage`](@ref), computed when a variable
first reads it, so a message nobody needs is never computed.

The streams are [Rocket.jl](https://github.com/ReactiveBayes/Rocket.jl) observables. Nothing runs
until something subscribes: a subscription to a marginal, or to the free energy, is what pulls the
computation through the graph. This is why a graph is first built and then *activated*, and why
the same graph serves streaming data, each observation propagating as it arrives (see
[Inference lifecycle](@ref concepts-inference-lifecycle)).

## [Messages and marginals](@id concepts-message-passing-types)

The engine wraps the two in their own types:

- [`Message`](@ref), a message along one edge, from a node towards a variable or back;
- [`Marginal`](@ref), a belief about a variable, formed from the product of the messages arriving
  at it, or about a cluster of a node's variables, computed by the node's marginal rule.

Both hold a distribution and forward its statistics, and record whether the value is clamped
(computed from constants and observations only) or initial (set before inference). A message
also carries its log scale, when log scales are tracked (see [Log scales](@ref lib-logscale)),
and both carry optional [annotations](@ref lib-annotations).

## [Next steps](@id concepts-message-passing-next)

- [Messages](@ref lib-message) and [Marginals](@ref lib-marginal): the types and their streams.
- [Inference lifecycle](@ref concepts-inference-lifecycle): construction, activation and
  observation.
- [`MessagePassingRulesBase`](https://reactivebayes.github.io/MessagePassingRulesBase.jl/dev/): how
  nodes, rules and algorithms are defined.
