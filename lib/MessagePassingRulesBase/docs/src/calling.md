```@meta
CurrentModule = MessagePassingRulesBase
```

# Calling rules

Any rule can be called directly, without a graph, which is how rules are tested and explored. A
call gives the inputs by name, resolves the rule as an engine would, runs it, and returns a
[`RuleResult`](@ref). [Inspecting rules](@ref) shows which rule a call would run without running
it. The examples on this page use a small node:

```@example calling
using MessagePassingRulesBase

struct Gauss   # a normal, by its mean and variance
    m::Float64
    v::Float64
end

struct Shift end   # out = in + c

@define_factor_node(node = Shift, type = Deterministic, interfaces = [:out, :in, :c])

@define_message_update_rule(
    node = Shift, target = :out, args = (m[:in]::Gauss, m[:c]::Real), logscale = 0,
    body = (args) -> Gauss(args.m[:in].m + args.m[:c], args.m[:in].v),
)

@define_message_update_rule(
    node = Shift, target = :in, args = (m[:out]::Gauss, m[:c]::Real), logscale = 0,
    body = (args) -> Gauss(args.m[:out].m - args.m[:c], args.m[:out].v),
)

result = @call_message_update_rule(node = Shift, target = :out, m = (in = Gauss(1.0, 2.0), c = 3.0))
```

## Calling a rule

Each kind of rule has a function, with the node and the target as positional arguments, and a
macro, with everything given by name. The inputs are `m` for messages and `q` for marginals,
`NamedTuple`s keyed by interface name, and `clusters` for joint marginals; the rest selects the
algorithm, supplies the context and collects the annotations. The context's services are not
checked (see [The rule context](@ref)).

```@docs
call_message_update_rule
call_marginal_update_rule
call_average_energy
@call_message_update_rule
@call_marginal_update_rule
@call_average_energy
```

## Reading the result

A [`RuleResult`](@ref) holds the result with everything that produced it. It shows itself in the
terminal as a report, one line per edge of the node with what it carried into the rule, and in a
notebook or on these pages as a card with the node drawn: the inputs as arrows in, messages solid
and marginals dashed, the target as the arrow out, with the result, its log scale, the rule that
ran and the other rules for the same target, as the result above shows.

```@example calling
getresult(result), getlogscale(result)
```

```@docs
RuleResult
getresult
getrule
MessagePassingRulesBase.getalgorithm
MessagePassingRulesBase.getcontext
MessagePassingRulesBase.getscratch
MessagePassingRulesBase.getarguments
MessagePassingRulesBase.gettarget
getannotations
```

A marginal rule whose cluster factorises returns a [`FactorizedCluster`](@ref) of its blocks, each
labelled with the members it covers; an engine hands each block to its members.

```@docs
FactorizedCluster
MessagePassingRulesBase.cluster_blocks
MessagePassingRulesBase.check_factorized_cluster
```

## When no rule fits

Every call throws a [`RuleNotFoundError`](@ref) when no rule fits, whose message diagnoses the
call and lists every rule for the node and target with, slot by slot, why it does not fit. Here
`c` is given as a `String`:

```@example calling
try
    @call_message_update_rule(node = Shift, target = :out, m = (in = Gauss(1.0, 2.0), c = "3"))
catch error
    showerror(stdout, error)
end
```

```@docs
MessagePassingRulesBase.RuleNotFoundError
MessagePassingRulesBase.RuleNotFound
```

## Resolving without the interactive layer

Code that builds its own [`RuleArgs`](@ref) resolves a rule with the `find_*` functions, which
never throw and return a [`RuleNotFound`](@ref) when nothing fits, and runs it with the
`message_passing_*` functions, which throw. Julia's dispatch does the resolution, over every
loaded package.

```@docs
MessagePassingRulesBase.find_message_rule
MessagePassingRulesBase.find_marginal_rule
MessagePassingRulesBase.find_average_energy
MessagePassingRulesBase.rule_algorithm
message_passing_rule
message_passing_rule!
message_passing_marginalrule
message_passing_marginalrule!
message_passing_average_energy
```

