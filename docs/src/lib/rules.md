# [Rules implementation](@id lib-rules)

## [Message update rules](@id lib-message-rules)

```@docs
rule
@rule
@call_rule
ReactiveMP.call_rule_make_node
ReactiveMP.call_rule_macro_parse_fn_args
ReactiveMP.call_rule_is_node_required
ReactiveMP.rule_macro_parse_on_tag
ReactiveMP.rule_macro_parse_fn_args
ReactiveMP.rule_macro_check_fn_args
```

## [Marginal update rules](@id lib-marginal-rules)


```@docs
marginalrule
@marginalrule
@call_marginalrule
```

## [Testing utilities for the update rules](@id lib-rules-tests)

```@docs
ReactiveMP.@test_rules
ReactiveMP.@test_marginalrules
```

## [Rule fallbacks](@id lib-rules-fallbacks)

```@docs
ReactiveMP.NodeFunctionRuleFallback
ReactiveMP.nodefunction
```

## [Table of available update rules](@id lib-rules-table)

!!! note
    The list below has been automatically generated with the `ReactiveMP.print_rules_table()` function.

```@eval
using ReactiveMP, Markdown
Markdown.parse(ReactiveMP.print_rules_table())
```