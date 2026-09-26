```@meta
CurrentModule = MessagePassingRulesBase
```

# The rule context

A rule reads the services its caller supplies from its `ctx` slot, a [`RuleContext`](@ref): the
random number generator, the engine's node, a matrix correction strategy, or a service of its
own. The context never takes part in dispatch. A rule declares the services it reads with the
`ctx` keyword, `ctx = (:rng,)`, and reads each as `ctx.name`; a service the context does not
supply reads as `nothing`.

```jldoctest context
julia> using MessagePassingRulesBase

julia> struct Scaled end

julia> @define_factor_node(node = Scaled, type = Deterministic, interfaces = [:out, :in])

julia> @define_message_update_rule(
           node = Scaled, target = :out, args = (m[:in]::Real,),
           ctx = (:scale,),
           body = (ctx, args) -> ctx.scale * args.m[:in],
       )

julia> ctx = MessagePassingRulesBase.RuleContext(scale = 3.0);

julia> getresult(@call_message_update_rule(node = Scaled, target = :out, m = (in = 2.0,), ctx = ctx))
6.0
```

```@docs
MessagePassingRulesBase.RuleContext
MessagePassingRulesBase.DEFAULT_CONTEXT_SERVICES
```

## Matrix correction

A rule that builds a matrix it must invert or factorise, such as a precision that must stay
positive definite, corrects it first. Which correction to apply is the caller's choice, a
MatrixCorrectionTools strategy supplied as `ctx.matrix_correction`, and each rule has its own
default for when none is set, which [`matrix_correction`](@ref) chooses between.

```@docs
matrix_correction
```

## Who checks the services

**An engine checks, a call by hand does not.** When an engine resolves a rule for a node in a
graph, it calls [`check_services`](@ref) once, before the rule ever runs, so a service nobody
supplies is an error that names the rule and the service, at setup. The calls by hand, the
`call_*` functions and macros and the `message_passing_*` functions, run the rule with whatever
context they are given, empty by default, and a declared service it lacks reads as `nothing`
inside the rule. That is what lets a test call a rule that reads `ctx.matrix_correction` without
supplying one: the rule then applies its own default. For the engine's guarantee in a call by
hand, check the rule first:

```jldoctest context
julia> spec = which_message_update_rule(Scaled, :out; m = (in = 2.0,));

julia> MessagePassingRulesBase.missing_services(spec, MessagePassingRulesBase.RuleContext())
(:scale,)

julia> MessagePassingRulesBase.check_services(spec, ctx)
```

```@docs
MessagePassingRulesBase.check_services
MessagePassingRulesBase.missing_services
```
