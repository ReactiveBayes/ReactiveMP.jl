```@meta
CurrentModule = MessagePassingRulesBase
```

# Inspecting rules

This page covers three questions: which [rule](@ref glossary-rule) a call would run, which rules
a node has, and whether the rules are consistent. A rule author asks them at the REPL, and a
rule package's tests ask them too.

The examples use the `Shift` node of [Calling rules](@ref), `out = in + c`, with rules towards
`out` and `in`:

```@setup inspecting
using MessagePassingRulesBase
struct Gauss; m::Float64; v::Float64; end
struct Shift end
@define_factor_node(node = Shift, type = Deterministic, interfaces = [:out, :in, :c])
@define_message_update_rule(node = Shift, target = :out, args = (m[:in]::Gauss, m[:c]::Real), logscale = 0, body = (args) -> Gauss(args.m[:in].m + args.m[:c], args.m[:in].v))
@define_message_update_rule(node = Shift, target = :in, args = (m[:out]::Gauss, m[:c]::Real), logscale = 0, body = (args) -> Gauss(args.m[:out].m - args.m[:c], args.m[:out].v))
```

## Which rule would run

The `which_*` queries resolve a rule without running it. They return the rule's
[`RuleSpec`](@ref):

```@example inspecting
which_message_update_rule(Shift, :in; m = (out = Gauss(4.0, 2.0), c = 3.0))
```

The `RuleSpec` shows the rule's inputs, its flags, its [log scale](@ref glossary-log-scale)
declaration, where it was defined, and its body.

```@docs
which_message_update_rule
which_marginal_update_rule
which_average_energy
@which_message_update_rule
@which_marginal_update_rule
@which_average_energy
MessagePassingRulesBase.RuleSpec
MessagePassingRulesBase.InputSpec
```

## Which rules exist

[`rule_coverage`](@ref) tabulates what a node can compute, and under which
[algorithm](@ref glossary-algorithm). Node packages show it on their pages.

```@example inspecting
MessagePassingRulesBase.rule_coverage(Shift)
```

`Shift` has a rule towards `out` and one towards `in`, both under `DefaultAlgorithm`. It has no
rule towards `c`. [`list_rules`](@ref) returns the rules themselves:

```@repl inspecting
MessagePassingRulesBase.list_rules(Shift, :in)
```

```@docs
MessagePassingRulesBase.list_rules
MessagePassingRulesBase.RuleCoverage
MessagePassingRulesBase.rule_coverage
MessagePassingRulesBase.visualize_spec
MessagePassingRulesBase.prettify_modules
```

## Checking the rules

A rule package's tests check its rules in two ways:

- against their nodes' declarations, with [`check_rules`](@ref);
- against each other, with [`check_rule_ambiguities`](@ref). Two rules that some call matches
  equally well make resolution throw a `MethodError`.

Each check takes the modules to check, every loaded module by default, and returns the problems
it finds. Both lists are empty for this page's module:

```@repl inspecting
MessagePassingRulesBase.check_rules(@__MODULE__)
MessagePassingRulesBase.check_rule_ambiguities(@__MODULE__)
```

```@docs
MessagePassingRulesBase.check_rules
MessagePassingRulesBase.RuleIssue
MessagePassingRulesBase.check_rule_ambiguities
MessagePassingRulesBase.duplicate_rules
```

## The registries

Each module that defines nodes or rules keeps a registry of what it defined. The registry is
filled when the module loads. It serves introspection only: listings, coverage, checks, and the
near misses a [`RuleNotFoundError`](@ref) reports. Resolution never reads it.

```@docs
MessagePassingRulesBase.Registry
MessagePassingRulesBase.registries
MessagePassingRulesBase.registered_rules
MessagePassingRulesBase.registered_nodes
MessagePassingRulesBase.registered_dependencies
```
