# [Callbacks](@id lib-callbacks)

Callbacks observe message passing. The engine reports an event before and after every rule call,
every product of messages, every form constraint, every marginal of a random variable, every
joint marginal a node computes, and every term of the free energy. A callback handler reacts to
the events it is interested in. Callbacks serve debugging, tracing and
monitoring, and they change nothing the engine computes. To transform the streams themselves, use
[stream postprocessors](@ref lib-stream-postprocessors).

## [Attaching callbacks](@id lib-callbacks-attach)

You give callbacks in two places:

- to a factor node, as the activation option `callbacks` of
  [`ReactiveMP.FactorNodeActivationOptions`](@ref). The node reports its message rule calls,
  [`ReactiveMP.BeforeMessageRuleCallEvent`](@ref) and
  [`ReactiveMP.AfterMessageRuleCallEvent`](@ref), its marginal rule calls, which compute the joint
  marginals of its clusters, and its term of the free energy,
  [`ReactiveMP.AfterFactorBoundFreeEnergyEvent`](@ref).
- to a variable, as the `callbacks` of the [`ReactiveMP.MessageProductContext`](@ref)s in its
  [`RandomVariableActivationOptions`](@ref). The variable reports the product and form
  constraint events, and the context of its marginal also reports the marginal events and the
  variable's term of the free energy, [`ReactiveMP.AfterVariableBoundEntropyEvent`](@ref).

An event after a rule call names the rule that ran, its `rule`: the
[`RuleSpec`](@extref MessagePassingRulesBase.RuleSpec), which shows where the rule is defined, the
node's rule fallback where no rule matched, or `nothing` where a `missing` input skipped the rule.
The engine builds an event only for a handler that listens to it, so a graph activated without
callbacks, or with callbacks for other events, runs as fast as one that reports nothing.

```@setup callbacks
using ReactiveMP, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, Rocket
import ReactiveMP: activate!, FactorNodeActivationOptions, MessageProductContext, get_stream_of_marginals
include(joinpath(pkgdir(ReactiveMP), "docs", "nodes.jl"))
```

The examples use the `Gaussian` node of [The example node](@ref example-node). The simplest
handler is a `NamedTuple` keyed by event names, each entry a function of the event. Mind the
trailing comma of a one-entry `NamedTuple`, `(name = f,)`.

```@example callbacks
x, y = randomvar(label = :x), datavar(label = :y)
prior = factornode(Gaussian, [(:out, x), (:μ, constvar(0.0)), (:v, constvar(10.0))])
likelihood = factornode(Gaussian, [(:out, y), (:μ, x), (:v, constvar(1.0))])

callbacks = (
    after_message_rule_call = (event) -> println("rule towards ", event.mapping.target, ": ", event.result),
    after_marginal_computation = (event) -> println("marginal: ", getdata(event.result)),
)

marginal_context = MessageProductContext(; callbacks)
activate!(x, RandomVariableActivationOptions(nothing, MessageProductContext(), marginal_context))
activate!(y, DataVariableActivationOptions())
foreach(node -> activate!(node, FactorNodeActivationOptions(; callbacks)), (prior, likelihood))

subscription = subscribe!(get_stream_of_marginals(x), (q) -> nothing)
new_observation!(y, 2.0)
unsubscribe!(subscription)
```

When the observation arrives, `x` reads the messages of both nodes, which runs their rules, and
forms its marginal from them.

A handler of a type of its own implements [`ReactiveMP.handle_event`](@ref) for each event it
reacts to. The engine builds an event only for a handler that [`ReactiveMP.listens`](@ref) to its
type, so a handler that reacts to a few events declares which, and the others cost it nothing.
The handler below counts the rule calls of each target:

```@example callbacks
struct RuleCounter
    counts::Dict{Any, Int}
end

ReactiveMP.listens(::RuleCounter, ::Type{<:ReactiveMP.Event}) = false
ReactiveMP.listens(::RuleCounter, ::Type{<:ReactiveMP.AfterMessageRuleCallEvent}) = true

function ReactiveMP.handle_event(counter::RuleCounter, event::ReactiveMP.AfterMessageRuleCallEvent)
    target = event.mapping.target
    counter.counts[target] = get(counter.counts, target, 0) + 1
    return nothing
end

counter = RuleCounter(Dict{Any, Int}())
x, y = randomvar(label = :x), datavar(label = :y)
prior = factornode(Gaussian, [(:out, x), (:μ, constvar(0.0)), (:v, constvar(10.0))])
likelihood = factornode(Gaussian, [(:out, y), (:μ, x), (:v, constvar(1.0))])
activate!(x, RandomVariableActivationOptions())
activate!(y, DataVariableActivationOptions())
foreach(node -> activate!(node, FactorNodeActivationOptions(; callbacks = counter)), (prior, likelihood))

subscription = subscribe!(get_stream_of_marginals(x), (q) -> nothing)
foreach(value -> new_observation!(y, value), (1.0, 2.0, 3.0))
unsubscribe!(subscription)
counter.counts
```

The prior's message towards `out` depends on constants alone, so it is computed once. The
likelihood's message towards `μ` is computed once per observation.
[`ReactiveMP.merge_callbacks`](@ref) combines several handlers into one.

```@docs
ReactiveMP.Event
ReactiveMP.event_name
ReactiveMP.invoke_callback
ReactiveMP.listens
ReactiveMP.handle_event
ReactiveMP.merge_callbacks
```

## [Event names](@id lib-callbacks-naming)

Every event is a concrete subtype of [`ReactiveMP.Event{E}`](@ref), where `E` is a `Symbol` that
names it. The struct of the event `:event_name` is `EventNameEvent`:

| Symbol | Struct |
|--------|--------|
| `:before_message_rule_call` | [`ReactiveMP.BeforeMessageRuleCallEvent`](@ref) |
| `:after_message_rule_call` | [`ReactiveMP.AfterMessageRuleCallEvent`](@ref) |
| `:before_product_of_two_messages` | [`ReactiveMP.BeforeProductOfTwoMessagesEvent`](@ref) |
| `:after_product_of_two_messages` | [`ReactiveMP.AfterProductOfTwoMessagesEvent`](@ref) |
| `:before_product_of_messages` | [`ReactiveMP.BeforeProductOfMessagesEvent`](@ref) |
| `:after_product_of_messages` | [`ReactiveMP.AfterProductOfMessagesEvent`](@ref) |
| `:before_form_constraint_applied` | [`ReactiveMP.BeforeFormConstraintAppliedEvent`](@ref) |
| `:after_form_constraint_applied` | [`ReactiveMP.AfterFormConstraintAppliedEvent`](@ref) |
| `:before_marginal_computation` | [`ReactiveMP.BeforeMarginalComputationEvent`](@ref) |
| `:after_marginal_computation` | [`ReactiveMP.AfterMarginalComputationEvent`](@ref) |
| `:before_marginal_rule_call` | [`ReactiveMP.BeforeMarginalRuleCallEvent`](@ref) |
| `:after_marginal_rule_call` | [`ReactiveMP.AfterMarginalRuleCallEvent`](@ref) |
| `:before_factor_bound_free_energy` | [`ReactiveMP.BeforeFactorBoundFreeEnergyEvent`](@ref) |
| `:after_factor_bound_free_energy` | [`ReactiveMP.AfterFactorBoundFreeEnergyEvent`](@ref) |
| `:before_variable_bound_entropy` | [`ReactiveMP.BeforeVariableBoundEntropyEvent`](@ref) |
| `:after_variable_bound_entropy` | [`ReactiveMP.AfterVariableBoundEntropyEvent`](@ref) |

Each event carries what happened in its fields, listed with it below.

## [Event spans](@id lib-callbacks-spans)

A "before" event and its "after" event share a `span_id`, from
[`ReactiveMP.generate_span_id`](@ref). A trace uses it to pair them, and to nest the products
inside the marginal they form.

```@docs
ReactiveMP.generate_span_id
```

## [All events](@id lib-callbacks-events)

```@docs
ReactiveMP.BeforeMessageRuleCallEvent
ReactiveMP.AfterMessageRuleCallEvent
ReactiveMP.BeforeProductOfTwoMessagesEvent
ReactiveMP.AfterProductOfTwoMessagesEvent
ReactiveMP.BeforeProductOfMessagesEvent
ReactiveMP.AfterProductOfMessagesEvent
ReactiveMP.BeforeFormConstraintAppliedEvent
ReactiveMP.AfterFormConstraintAppliedEvent
ReactiveMP.BeforeMarginalComputationEvent
ReactiveMP.AfterMarginalComputationEvent
ReactiveMP.BeforeMarginalRuleCallEvent
ReactiveMP.AfterMarginalRuleCallEvent
ReactiveMP.BeforeFactorBoundFreeEnergyEvent
ReactiveMP.AfterFactorBoundFreeEnergyEvent
ReactiveMP.BeforeVariableBoundEntropyEvent
ReactiveMP.AfterVariableBoundEntropyEvent
```
