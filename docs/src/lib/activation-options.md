# [Activation options](@id lib-activation-options)

A factor node is activated with a [`ReactiveMP.FactorNodeActivationOptions`](@ref), which says
what its rules run under. Every option is a keyword with a default, so a node that needs nothing
special is activated with `FactorNodeActivationOptions()`. RxInfer builds these options from a
model's settings; with the engine alone, they are given to [`ReactiveMP.activate!`](@ref) directly.

```@docs
ReactiveMP.FactorNodeActivationOptions
```

The options that concern the node's rules are described here. The others observe or transform
what the node computes: `callbacks` ([Callbacks](@ref lib-callbacks)), `postprocessor`
([Stream postprocessors](@ref lib-stream-postprocessors)) and `annotations`
([Annotations](@ref lib-annotations)).

## [The algorithm](@id lib-activation-options-algorithm)

A node's rules belong to algorithms, and `algorithm` chooses the one the node runs under. It is
`nothing` by default, the node's own,
[`default_algorithm`](@extref MessagePassingRulesBase.default_algorithm), which is
[`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm)`()` for most nodes. A node
whose algorithm carries settings, such as the approximation method of the Delta node,
[`DeltaApproximation`](@extref DeltaMessagePassingRules.DeltaApproximation), or that declares no
default, such as the autoregressive node,
[`ARVMP`](@extref AutoregressiveMessagePassingRules.ARVMP), is given one:

```julia
activate!(node, FactorNodeActivationOptions(; algorithm = DeltaApproximation(method = Unscented())))
```

The algorithm also decides what each rule reads, when it declares its own dependencies, and
[`bethe_free_energy`](@ref) takes the same algorithm per node, for the average energies.

## [Services: the context](@id lib-activation-options-context)

A rule reads the services it needs, such as a random number generator, from its context, `ctx`,
and declares them, `ctx = (:rng,)`. The engine supplies three to every node's rules, with
[`ReactiveMP.node_context`](@ref): `node`, the factor node; `rng`, the task's random number
generator; and `matrix_correction`, `nothing`, so each rule applies its own. The option `context`,
a `NamedTuple`, is merged over them: it adds services and overrides the engine's.

```@docs
ReactiveMP.node_context
```

A rule declaring a service that neither supplies is an error when the rule is resolved, before it
runs, naming the rule and the service.

```@setup lib-activation-options
using ReactiveMP, MessagePassingRulesBase, BayesBase, Rocket
import ReactiveMP: activate!, FactorNodeActivationOptions, get_stream_of_marginals
```

Here a toy node scales its input by a service the engine does not supply, `scale`:

```@example lib-activation-options
struct Scale end

@define_factor_node(node = Scale, type = Deterministic, interfaces = [:out, :in])

@define_message_update_rule(
    node = Scale, target = :out, args = (m[:in]::PointMass,), ctx = (:scale,),
    body = (ctx, args) -> PointMass(ctx.scale * mean(args.m[:in])),
)

x, y = datavar(), randomvar()
node = factornode(Scale, [(:out, y), (:in, x)])
activate!(x, DataVariableActivationOptions())
activate!(y, RandomVariableActivationOptions())
activate!(node, FactorNodeActivationOptions(; context = (scale = 3.0,)))

subscription = subscribe!(get_stream_of_marginals(y), (q) -> println("q(y) = ", getdata(q)))
new_observation!(x, 2.0)
unsubscribe!(subscription)
```

[`bethe_free_energy`](@ref) runs the nodes' average energies with the engine's context alone: the
services of `context` do not reach them.

## [Rule fallbacks](@id lib-activation-options-rulefallback)

When no rule matches a message's inputs, the message is a
[`RuleNotFoundError`](@extref MessagePassingRulesBase.RuleNotFoundError), which lists the near
misses. The option `rulefallback` gives a message instead, only where no rule matches: it never
replaces a rule, and an error inside a rule propagates. It is called as
`rulefallback(fform, target, args)`; returning `nothing` is the error again.
[`NodeFunctionRuleFallback`](@extref MessagePassingRulesBase.NodeFunctionRuleFallback)`()` gives
the message of the node's function against its inputs, as an unnormalised log-density:

```julia
activate!(node, FactorNodeActivationOptions(; rulefallback = NodeFunctionRuleFallback()))
```

A message a fallback computed has an undefined log scale. A marginal rule and an average energy
have no fallback.

## [Log scales](@id lib-activation-options-logscales)

With `logscales = true`, the node's messages carry the log scale each rule declares, and the rules
that read their inputs' log scales receive them; the variables combine them through their
products. It is off by default: the messages carry `nothing`, and a rule that reads its inputs'
log scales is an error naming the option. A graph tracks log scales when all its nodes do; see
[Log scales](@ref lib-logscale).

## [Diagnostics](@id lib-activation-options-diagnostics)

Three audits of the rules a node runs, all off by default, are set with `diagnostics`:
`check_everything_pure` stops at an impure rule, `check_everything_inplace` reports once each rule
with no in-place form, and `checked_buffers` poisons the memory the engine recycles before each
reuse, so that a rule reading its scratch before writing it shows `NaN`. Each names the rule it
objects to, by its node, target, algorithm and the place it is defined. Purity is declared, not
proved: the audit reads what the rule and its algorithm declare (see
[`ispure`](@extref MessagePassingRulesBase.ispure)).

```julia
activate!(node, FactorNodeActivationOptions(; diagnostics = EngineDiagnostics(check_everything_pure = true)))
```

```@docs
ReactiveMP.EngineDiagnostics
ReactiveMP.ImpureRuleError
```

## [Variables](@id lib-activation-options-variables)

Variables have their own options, positional structs: a random variable's
[`RandomVariableActivationOptions`](@ref), with the product contexts of its messages and of its
marginal, and a data variable's [`DataVariableActivationOptions`](@ref). They are described with
the [variables](@ref lib-variables).
