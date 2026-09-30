```@meta
CurrentModule = MessagePassingRulesBase
```

# Calling rules

You can call any [rule](@ref glossary-rule) directly, without a graph. This is how rules are
tested and explored. A call gives the inputs by name, resolves the rule as an engine would, runs
it, and returns a [`RuleResult`](@ref). [Inspecting rules](@ref) shows which rule a call would
run, without running it.

The examples on this page use a small [deterministic node](@ref glossary-deterministic-node),
`out = in + c`, with a normal type of its own:

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

Each kind of rule has a function and a macro. The function takes the node and the target as
positional arguments. The macro takes everything by name.

The inputs are three keywords:

- `m`, the [messages](@ref glossary-message), a `NamedTuple` keyed by interface name;
- `q`, the [marginals](@ref glossary-marginal), keyed the same way;
- `clusters`, the joint marginals.

The other keywords select the [algorithm](@ref glossary-algorithm), supply the
[`RuleContext`](@ref) and collect the annotations. A call does not check the
context's services (see [The rule context](@ref)). The
[Keyword reference](@ref keyword-reference) lists every keyword.

```@docs
call_message_update_rule
call_marginal_update_rule
call_average_energy
@call_message_update_rule
@call_marginal_update_rule
@call_average_energy
```

## Reading the result

A [`RuleResult`](@ref) holds the result together with everything that produced it. On these
pages, and in a notebook, it shows itself as a card with the node drawn, as the result above
does:

- the inputs are arrows in, solid for messages and dashed for marginals;
- the target is the arrow out;
- below the drawing are the result, its [log scale](@ref glossary-log-scale), the rule that ran
  and the other rules for the same target.

In the terminal, it shows itself as a report with one line per edge of the node, giving what the
edge carried into the rule. The accessors below read its parts:

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

A marginal rule whose cluster factorises returns a [`FactorizedCluster`](@ref) of its blocks.
Each block is labelled with the members it covers, and an engine hands each block to its
members.

```@docs
FactorizedCluster
MessagePassingRulesBase.cluster_blocks
MessagePassingRulesBase.check_factorized_cluster
```

## When no rule fits

Every call throws a [`RuleNotFoundError`](@ref) when no rule fits. Its message diagnoses the
call. It lists every rule for the node and target and says, slot by slot, why each one does not
fit. Here `c` is given as a `String`:

```@example calling
try
    @call_message_update_rule(node = Shift, target = :out, m = (in = Gauss(1.0, 2.0), c = "3"))
catch err
    showerror(stdout, err)
end
```

```@docs
MessagePassingRulesBase.RuleNotFoundError
MessagePassingRulesBase.RuleNotFound
MessagePassingRulesBase.rule_not_found_hint
```

## Resolving without the interactive layer

Code that builds its own [`RuleArgs`](@ref) resolves a rule with the `find_*` functions. They
never throw: they return a [`RuleNotFound`](@ref) when nothing fits. The code then runs the rule
with the `message_passing_*` functions, which do throw. Julia's dispatch does the resolution,
over every loaded package.

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

## For tools

A tool such as the test tooling calls rules the way the interactive functions do. It reads its
inputs with the same functions, and it can observe which rule each call selects.

```@docs
MessagePassingRulesBase.as_target
MessagePassingRulesBase.as_cluster
MessagePassingRulesBase.interactive_args
MessagePassingRulesBase.add_selection_observer!
```
