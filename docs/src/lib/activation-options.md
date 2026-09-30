# [Activation options](@id lib-activation-options)

You activate a factor node with a [`ReactiveMP.FactorNodeActivationOptions`](@ref), which says
what its rules run under. Every option is a keyword with a default, so a node that needs nothing
special is activated with `FactorNodeActivationOptions()`. RxInfer builds these options from a
model's settings. With the engine alone, you pass them to [`ReactiveMP.activate!`](@ref).

```@docs
ReactiveMP.FactorNodeActivationOptions
ReactiveMP.getcallbacks
ReactiveMP.getpostprocessor
```

This page describes the options that concern the node's rules. The others observe or transform
what the node computes: `callbacks` ([Callbacks](@ref lib-callbacks)), `postprocessor`
([Stream postprocessors](@ref lib-stream-postprocessors)) and `annotations`
([Annotations](@ref lib-annotations)).

```@setup options
using ReactiveMP, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, Rocket
import ReactiveMP: activate!, FactorNodeActivationOptions, get_stream_of_marginals
include(joinpath(pkgdir(ReactiveMP), "docs", "nodes.jl"))
```

The examples use the `Gaussian` node of [The example node](@ref example-node). Each builds the
same graph, a latent `x` with a normal prior observed through a second node, and activates the
second node with the option it shows:

```@example options
function observe(node_type, options; observation = 2.0)
    x, y = randomvar(label = :x), datavar(label = :y)
    prior = factornode(Gaussian, [(:out, x), (:μ, constvar(0.0)), (:v, constvar(10.0))])
    likelihood = factornode(node_type, [(:out, y), (:μ, x), (:v, constvar(1.0))])
    activate!(x, RandomVariableActivationOptions())
    activate!(y, DataVariableActivationOptions())
    activate!(prior, FactorNodeActivationOptions())
    activate!(likelihood, options)
    posteriors = Marginal[]
    subscription = subscribe!(get_stream_of_marginals(x), (q) -> push!(posteriors, q))
    new_observation!(y, observation)
    unsubscribe!(subscription)
    return last(posteriors)
end

observe(Gaussian, FactorNodeActivationOptions())
```

## [The algorithm](@id lib-activation-options-algorithm)

A node's rules belong to [algorithms](@extref MessagePassingRulesBase glossary-algorithm), and
`algorithm` chooses the one the node runs under. Its default, `nothing`, is the node's own,
[`default_algorithm`](@extref MessagePassingRulesBase.default_algorithm), which is
[`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm)`()` for most nodes.

An algorithm may carry settings. The algorithm below extends the default and inflates the
variance of the message towards `μ` by a factor:

```@example options
struct Inflated{T} <: DefaultAlgorithmExtension
    factor::T
end

@define_message_update_rule(
    node = Gaussian, target = :μ, algorithm = Inflated,
    args = (m[:out]::PointMass, m[:v]::PointMass), logscale = 0,
    body = (algo, args) -> NormalMeanVariance(mean(args.m[:out]), algo.factor * mean(args.m[:v])),
)

observe(Gaussian, FactorNodeActivationOptions(; algorithm = Inflated(3.0)))
```

The observation counts as one with variance 3, so the posterior precision is
``1/10 + 1/3``. Rules the extension does not override are the default's.

Some nodes need an algorithm with settings. The Delta node takes its approximation method in
[`DeltaApproximation`](@extref DeltaMessagePassingRules.DeltaApproximation), and the
autoregressive node declares no default, so a model gives it
[`ARVMP`](@extref AutoregressiveMessagePassingRules.ARVMP). The algorithm also decides what each
rule reads, when it declares its own dependencies. [`bethe_free_energy`](@ref) takes the same
algorithm per node, for the average energies.

## [Services: the context](@id lib-activation-options-context)

A rule reads the [services](@extref MessagePassingRulesBase glossary-service) it needs, such as a
random number generator, from its context, `ctx`, and declares them, `ctx = (:rng,)`. The engine
supplies three to every node's rules, with [`ReactiveMP.node_context`](@ref):

- `node`, the factor node;
- `rng`, the task's random number generator;
- `matrix_correction`, `nothing`, so that each rule applies its own.

The option `context`, a `NamedTuple`, is merged over them: it adds services and overrides the
engine's.

```@docs
ReactiveMP.node_context
```

Here a deterministic node scales its input by a service the engine does not supply, `scale`:

```@example options
struct Scale end

@define_factor_node(node = Scale, type = Deterministic, interfaces = [:out, :in])

@define_message_update_rule(
    node = Scale, target = :out, args = (m[:in]::PointMass,), ctx = (:scale,),
    body = (ctx, args) -> PointMass(ctx.scale * mean(args.m[:in])),
)

input, output = datavar(label = :input), randomvar(label = :output)
scale = factornode(Scale, [(:out, output), (:in, input)])
activate!(input, DataVariableActivationOptions())
activate!(output, RandomVariableActivationOptions())
activate!(scale, FactorNodeActivationOptions(; context = (scale = 3.0,)))

subscription = subscribe!(get_stream_of_marginals(output), (q) -> println("q(output) = ", q))
new_observation!(input, 2.0)
unsubscribe!(subscription)
```

A rule that declares a service nobody supplies is an error when the engine resolves the rule,
before it runs. The error names the rule and the service.
[`bethe_free_energy`](@ref) runs the average energies with the engine's context alone: the
services of `context` do not reach them.

## [Rule fallbacks](@id lib-activation-options-rulefallback)

When no rule matches a message's inputs, the engine throws a
[`RuleNotFoundError`](@extref MessagePassingRulesBase.RuleNotFoundError), which lists the near
misses. The node below is a normal density given by a function, and it has no rules:

```@example options
noisy(μ, v) = NormalMeanVariance(μ, v)

@define_factor_node(node = noisy, type = Stochastic, interfaces = [:out, :μ, :v])

try
    observe(noisy, FactorNodeActivationOptions())
catch err
    showerror(stdout, err)
end
```

The option `rulefallback` gives a message instead, only where no rule matches. It never replaces
a rule, and an error inside a rule propagates. The engine calls it as
`rulefallback(fform, target, args)`, and a return value of `nothing` restores the error.
[`NodeFunctionRuleFallback`](@extref MessagePassingRulesBase.NodeFunctionRuleFallback)`()` gives
the node's log-density as a function of the target, an unnormalised message:

```@example options
q = observe(noisy, FactorNodeActivationOptions(; rulefallback = NodeFunctionRuleFallback()))
exact = NormalMeanVariance(2.0 / 1.1, 1 / 1.1)
logpdf(q, 1.0) - logpdf(q, 0.0), logpdf(exact, 1.0) - logpdf(exact, 0.0)
```

The posterior is the product of the prior's message and the fallback's, a `BayesBase.ProductOf`
with no closed form. Its log-density agrees with the exact posterior's up to a constant. A
[form constraint](@ref custom-functional-form) turns such a product into a distribution. A
message a fallback computed has an undefined log scale. A marginal rule and an average energy
have no fallback.

## [Initial messages](@id lib-activation-options-initial-messages)

A rule that reads the message on its own edge needs that message before the first update. A node
may declare one, [`initial_messages`](@extref MessagePassingRulesBase.initial_messages), which the
engine sets on its inbound message wherever nothing is set yet. The option `initial_messages`
sets them for one node: a `NamedTuple` keyed by interface name or alias, with a tuple of one
message per member for a group. It replaces the node's declared message and any message set on
that edge before. An interface on a constant is left alone.

The option concerns the node's own edge only. Here two `Gaussian` nodes share the variable `x`,
and only the first is given a message on its `μ` edge:

```@example options
x = randomvar(label = :x)
y1, y2 = datavar(label = :y1), datavar(label = :y2)
first_node = factornode(Gaussian, [(:out, y1), (:μ, x), (:v, constvar(1.0))])
second_node = factornode(Gaussian, [(:out, y2), (:μ, x), (:v, constvar(1.0))])
activate!(x, RandomVariableActivationOptions())
foreach(y -> activate!(y, DataVariableActivationOptions()), (y1, y2))
activate!(first_node, FactorNodeActivationOptions(; initial_messages = (μ = NormalMeanVariance(0.0, 100.0),)))
activate!(second_node, FactorNodeActivationOptions())

on_μ(node) = Rocket.getrecent(ReactiveMP.get_stream_of_inbound_messages(ReactiveMP.getinterface(node, 2)))
on_μ(first_node), on_μ(second_node)
```

The first node's `μ` edge starts with the message, and the second node's has none yet. In RxInfer
this is `where { initial_messages = (μ = …,) }` on the node. `@initialization μ(x) = …` sets the
message on every edge of `x` instead.

## [Log scales](@id lib-activation-options-logscales)

With `logscales = true`, the node's messages carry the
[log scale](@extref MessagePassingRulesBase glossary-log-scale) each rule declares. The rules that
read their inputs' log scales receive them, and the variables combine them through their
products. A graph tracks log scales when all its nodes do. Here only the second node does:

```@example options
try
    getlogscale(observe(Gaussian, FactorNodeActivationOptions(; logscales = true)))
catch err
    showerror(stdout, err)
end
```

The prior's message carries `nothing`, and so does the posterior, the product of the two
messages. [Log scales](@ref lib-logscale) activates every node with the option and reads the log
evidence from the posterior. The option is off by default: messages carry `nothing`, and a rule
that reads its inputs' log scales is an error naming the option.

## [Diagnostics](@id lib-activation-options-diagnostics)

The option `diagnostics`, an [`ReactiveMP.EngineDiagnostics`](@ref), sets three audits of the
rules a node runs, all off by default:

- `check_everything_pure` stops at an impure rule;
- `check_everything_inplace` reports once each rule with no
  [in-place](@extref MessagePassingRulesBase glossary-in-place-rule) form;
- `checked_buffers` fills the memory the engine recycles with `NaN` before each reuse, so that a
  rule reading its [scratch](@extref MessagePassingRulesBase glossary-scratch) before writing it
  returns `NaN`.

Each audit names the rule it objects to, by its node, target, algorithm and the place it is
defined. Purity is declared, not proved: the audit reads what the rule and its algorithm declare
(see [`ispure`](@extref MessagePassingRulesBase.ispure)). The rule below draws from the random
number generator and declares itself impure:

```@example options
struct Jitter end

@define_factor_node(node = Jitter, type = Stochastic, interfaces = [:out, :μ, :v])

@define_message_update_rule(
    node = Jitter, target = :μ, args = (m[:out]::PointMass, m[:v]::PointMass),
    ctx = (:rng,), pure = false,
    body = (ctx, args) -> NormalMeanVariance(mean(args.m[:out]) + 0.1 * randn(ctx.rng), mean(args.m[:v])),
)

try
    observe(Jitter, FactorNodeActivationOptions(; diagnostics = ReactiveMP.EngineDiagnostics(check_everything_pure = true)))
catch err
    showerror(stdout, err)
end
```

```@docs
ReactiveMP.EngineDiagnostics
ReactiveMP.ImpureRuleError
```

## [Variables](@id lib-activation-options-variables)

Variables have options of their own, positional structs. A random variable's
[`RandomVariableActivationOptions`](@ref) hold its stream postprocessor and the product contexts
of its messages and of its marginal. A data variable's [`DataVariableActivationOptions`](@ref)
ask for its prediction or link it to other variables. The [Variables](@ref lib-variables) page
describes both.
