```@meta
CurrentModule = MessagePassingRulesBase
```

# Internals

This page is for engine authors and contributors. It lists the calls an engine makes to run a
resolved rule, and the internal helpers a contributor needs.

## The engine interface

An engine uses a rule in three steps:

1. It resolves the rule once, when it sets a node up, with [`find_message_rule`](@ref) and its
   siblings.
2. It checks the rule with [`check_services`](@ref) and [`check_reads_logscale`](@ref).
3. It runs the rule on every update with [`execute_rule`](@ref). `execute_rule` builds no
   [`RuleResult`](@ref), so it allocates nothing beyond the rule's own work.

These names are public, for engine authors. A rule author never calls them.

```@docs
MessagePassingRulesBase.execute_rule
MessagePassingRulesBase.execute_rule_with_logscale
MessagePassingRulesBase.rule_scratch
MessagePassingRulesBase.rule_scratch_type
MessagePassingRulesBase.check_reads_logscale
```

## Internal helpers

These names are not part of the public API. The macros' docstrings share text through `const`
string fragments, `DOC_RULE_*`, `DOC_CALL_*`, `DOC_MPR_*` and `DOC_DEPENDENCY_ENTRIES`. The
fragments have no docstrings of their own.

```@docs
MessagePassingRulesBase.default_inputs_match
MessagePassingRulesBase.WithLogScale
MessagePassingRulesBase.FromBody
```
