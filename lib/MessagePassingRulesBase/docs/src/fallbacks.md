```@meta
CurrentModule = MessagePassingRulesBase
```

# Rule fallbacks

Where no rule fits, an engine may consult a **rule fallback** instead of reporting the error: a
callable taking the node, the target and the rule's arguments, `fallback(node, target, args)`,
and returning a message, or `nothing` where it has none either. It is consulted only when
resolution returns a [`RuleNotFound`](@ref), so it never replaces a rule that exists, and an
error inside a rule is never turned into a fallback. ReactiveMP takes it as the activation option
`rulefallback`. A message a fallback computes has an undefined log scale.

[`NodeFunctionRuleFallback`](@ref) is the one this package provides: for a stochastic node
without groups, the node's log-density in the target, every other input collapsed to a point.
Its message is a [`NodeFunctionLogPdf`](@ref), unnormalised, which a form constraint or a product
with a proper distribution turns into a distribution.

```jldoctest fallbacks
julia> using MessagePassingRulesBase

julia> struct Plain end

julia> @define_factor_node(node = Plain, type = Deterministic, interfaces = [:out, :in])

julia> NodeFunctionRuleFallback()(Plain, MessagePassingRulesBase.Target(:out), MessagePassingRulesBase.RuleArgs(m = (in = 1.0,))) === nothing
true
```

A deterministic node has no log-density, so the fallback has nothing for it.

```@docs
NodeFunctionRuleFallback
MessagePassingRulesBase.NodeFunctionLogPdf
```
