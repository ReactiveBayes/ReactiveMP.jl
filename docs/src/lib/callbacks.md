# [Callbacks](@id lib-callbacks)

Callbacks observe the message passing procedure: the engine reports an event before and after
every rule call, every product of messages, every form constraint and every marginal of a random
variable, and a callback handler reacts to the events it is interested in. They serve debugging,
tracing and monitoring, and change nothing the engine computes; to transform the streams
themselves, use [stream postprocessors](@ref lib-stream-postprocessors).

## [Attaching callbacks](@id lib-callbacks-attach)

Callbacks are given in two places:

- to a factor node, as the activation option `callbacks` of
  [`ReactiveMP.FactorNodeActivationOptions`](@ref): the rule call events,
  [`ReactiveMP.BeforeMessageRuleCallEvent`](@ref) and [`ReactiveMP.AfterMessageRuleCallEvent`](@ref);
- to a variable, as the `callbacks` of the [`ReactiveMP.MessageProductContext`](@ref)s in its
  [`RandomVariableActivationOptions`](@ref): the product and form constraint events, and, for the
  context of the marginal, the marginal events.

The simplest handler is a `NamedTuple` keyed by event names, each entry a function of the event;
mind its trailing comma, `(name = f,)`:

```@setup callbacks
using ReactiveMP, StandardMessagePassingRules, BayesBase, ExponentialFamily, Rocket
import ReactiveMP: activate!, FactorNodeActivationOptions, MessageProductContext, get_stream_of_marginals
```

```@example callbacks
x, y = randomvar(), datavar()
prior = factornode(NormalMeanVariance, [(:out, x), (:μ, constvar(0.0)), (:v, constvar(1.0))])
likelihood = factornode(NormalMeanVariance, [(:out, y), (:μ, x), (:v, constvar(1.0))])

callbacks = (
    after_message_rule_call = (event) -> println("rule towards ", event.mapping.target, ": ", event.result),
    after_marginal_computation = (event) -> println("marginal: ", getdata(event.result)),
)

marginal_context = MessageProductContext(; callbacks)
activate!(x, RandomVariableActivationOptions(nothing, MessageProductContext(), marginal_context))
activate!(y, DataVariableActivationOptions())
foreach(node -> activate!(node, FactorNodeActivationOptions(; callbacks)), (prior, likelihood))

subscription = subscribe!(get_stream_of_marginals(x), (q) -> nothing)
new_observation!(y, 1.0)
unsubscribe!(subscription)
```

A handler of a type of its own implements [`ReactiveMP.handle_event`](@ref) for each event it
reacts to, and [`ReactiveMP.merge_callbacks`](@ref) combines several handlers into one.

```@docs
ReactiveMP.Event
ReactiveMP.event_name
ReactiveMP.invoke_callback
ReactiveMP.handle_event
ReactiveMP.merge_callbacks
```

## [Event names](@id lib-callbacks-naming)

Every event is a concrete subtype of [`ReactiveMP.Event{E}`](@ref), `E` being a `Symbol` that
names it, and the struct of the event `:event_name` is `EventNameEvent`:

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

Each event carries what happened in its fields, listed with it below.

## [Event spans](@id lib-callbacks-spans)

A "before" event and its "after" event share a `span_id`, from
[`ReactiveMP.generate_span_id`](@ref), so that a trace can pair them, and nest the products inside
the marginal they form.

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
```
