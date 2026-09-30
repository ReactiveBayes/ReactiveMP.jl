```@meta
CurrentModule = MessagePassingRulesBase
```

# The rule context

A rule reads the [services](@ref glossary-service) its caller supplies from its `ctx` slot, a
[`RuleContext`](@ref). A service is a value the rule needs from whoever runs it: the random
number generator, the engine's node, a matrix correction strategy, or a service of the rule's
own. The context never takes part in dispatch.

A rule declares the services it reads with the `ctx` keyword, as `ctx = (:rng,)`, and reads each
one as `ctx.name`. A service the context does not supply reads as `nothing`.

```@example context
using MessagePassingRulesBase

struct Scaled end   # out = scale · in, with scale a service

@define_factor_node(node = Scaled, type = Deterministic, interfaces = [:out, :in])

@define_message_update_rule(
    node = Scaled, target = :out, args = (m[:in]::Real,),
    ctx = (:scale,),
    body = (ctx, args) -> ctx.scale * args.m[:in],
)

ctx = MessagePassingRulesBase.RuleContext(scale = 3.0)

@call_message_update_rule(node = Scaled, target = :out, m = (in = 2.0,), ctx = ctx)
```

The rule reads `scale` from the context it is given, and returns `6.0`.

```@docs
MessagePassingRulesBase.RuleContext
MessagePassingRulesBase.DEFAULT_CONTEXT_SERVICES
```

## Matrix correction

Some rules build a matrix they must invert or factorise, such as a precision matrix that must
stay positive definite. Such a rule corrects the matrix first.

The caller chooses the correction. It supplies a MatrixCorrectionTools strategy as
`ctx.matrix_correction`. Each rule has its own default for when none is set, and
[`matrix_correction`](@ref) chooses between the two.

```@docs
matrix_correction
```

## Who checks the services

**An engine checks the services. A call by hand does not.**

When an engine resolves a rule for a node in a graph, it calls [`check_services`](@ref) once,
before the rule ever runs. A service that nobody supplies is then an error at setup, and the
error names the rule and the service.

The calls by hand are the `call_*` functions and macros and the `message_passing_*` functions.
They run the rule with whatever context they are given, which is empty by default. A declared
service that the context lacks reads as `nothing` inside the rule. A test can therefore call a
rule that reads `ctx.matrix_correction` without supplying one, and the rule applies its own
default.

To get the engine's guarantee in a call by hand, check the rule first. The rule's
[`RuleSpec`](@ref) lists the services it declares:

```@example context
spec = which_message_update_rule(Scaled, :out; m = (in = 2.0,))
```

```@repl context
MessagePassingRulesBase.missing_services(spec, MessagePassingRulesBase.RuleContext())
MessagePassingRulesBase.check_services(spec, ctx)
```

An empty context lacks `scale`. The context `ctx` above supplies it, so
[`check_services`](@ref) returns `nothing`. For the empty context, it throws the error an engine
reports at setup:

```@example context
try
    MessagePassingRulesBase.check_services(spec, MessagePassingRulesBase.RuleContext())
catch err
    showerror(stdout, err)
end
```

```@docs
MessagePassingRulesBase.check_services
MessagePassingRulesBase.missing_services
```
