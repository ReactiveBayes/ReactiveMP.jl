# [Rules implementation](@id lib-rules)

## [Message update rules](@id lib-message-rules)

```@docs
rule
@rule
@call_rule
ReactiveMP.print_rules_table
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

## [Table of available update rules](@id lib-rules-table)

```@example tables
using ReactiveMP, Markdown
Markdown.parse(ReactiveMP.print_rules_table())
```