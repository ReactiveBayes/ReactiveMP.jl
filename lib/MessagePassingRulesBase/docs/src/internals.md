```@meta
CurrentModule = MessagePassingRulesBase
```

# Internals

## The engine interface

An engine resolves a rule once, when it sets a node up, with [`find_message_rule`](@ref) and its
siblings, checks it with [`check_services`](@ref) and [`check_reads_logscale`](@ref), and then
runs it on every update with [`execute_rule`](@ref), which builds no [`RuleResult`](@ref) and so
allocates nothing beyond the rule's own work. These names are public, for engine authors; a rule
author never calls them.

```@docs
MessagePassingRulesBase.execute_rule
MessagePassingRulesBase.execute_rule_with_logscale
MessagePassingRulesBase.rule_scratch
MessagePassingRulesBase.check_reads_logscale
```

## Internal helpers

Not part of the public API. The macros' shared docstring text is held in `const` string
fragments, `DOC_RULE_*`, `DOC_CALL_*`, `DOC_MPR_*` and `DOC_DEPENDENCY_ENTRIES`, which are not
documented themselves.

```@docs
MessagePassingRulesBase.default_inputs_match
MessagePassingRulesBase.WithLogScale
MessagePassingRulesBase.FromBody
```
