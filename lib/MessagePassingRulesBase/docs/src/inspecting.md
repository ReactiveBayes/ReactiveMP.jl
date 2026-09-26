```@meta
CurrentModule = MessagePassingRulesBase
```

# Inspecting rules

Which rule a call would run, which rules a node has, and whether they are consistent: the
queries a rule author uses at the REPL and a rule package's tests run. The examples use the
`Shift` node of [Calling rules](@ref), `out = in + c`, with rules towards `out` and `in`.

```@setup inspecting
using MessagePassingRulesBase
struct Gauss; m::Float64; v::Float64; end
struct Shift end
@define_factor_node(node = Shift, type = Deterministic, interfaces = [:out, :in, :c])
@define_message_update_rule(node = Shift, target = :out, args = (m[:in]::Gauss, m[:c]::Real), logscale = 0, body = (args) -> Gauss(args.m[:in].m + args.m[:c], args.m[:in].v))
@define_message_update_rule(node = Shift, target = :in, args = (m[:out]::Gauss, m[:c]::Real), logscale = 0, body = (args) -> Gauss(args.m[:out].m - args.m[:c], args.m[:out].v))
```

## Which rule would run

The `which_*` queries resolve without running, and return the [`RuleSpec`](@ref), which shows its
inputs, its flags, its log-scale declaration, where it was defined and its body:

```@repl inspecting
which_message_update_rule(Shift, :in; m = (out = Gauss(4.0, 2.0), c = 3.0))
```

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

[`rule_coverage`](@ref) tabulates what a node can compute, under which algorithm; node packages
show it on their pages:

```@repl inspecting
MessagePassingRulesBase.rule_coverage(Shift)
```

[`list_rules`](@ref) returns the rules themselves:

```@repl inspecting
MessagePassingRulesBase.list_rules(Shift, :in)
```

```@docs
MessagePassingRulesBase.list_rules
MessagePassingRulesBase.RuleCoverage
MessagePassingRulesBase.rule_coverage
MessagePassingRulesBase.visualize_spec
```

## Checking the rules

A rule package's tests check its rules against their nodes' declarations with
[`check_rules`](@ref), and against each other with [`check_rule_ambiguities`](@ref): a pair of
rules some call matches equally well makes resolution throw a `MethodError`.

```@repl inspecting
MessagePassingRulesBase.check_rules(), MessagePassingRulesBase.check_rule_ambiguities()
```

```@docs
MessagePassingRulesBase.check_rules
MessagePassingRulesBase.RuleIssue
MessagePassingRulesBase.check_rule_ambiguities
MessagePassingRulesBase.duplicate_rules
```

## The registries

Each module that defines nodes or rules keeps a registry of what it defined, filled when the
module loads. It is for introspection only, listings, coverage, checks and the near misses of a
[`RuleNotFoundError`](@ref); resolution never reads it.

```@docs
MessagePassingRulesBase.Registry
MessagePassingRulesBase.registries
MessagePassingRulesBase.registered_rules
MessagePassingRulesBase.registered_nodes
MessagePassingRulesBase.registered_dependencies
```
