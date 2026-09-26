# Logic

Deterministic nodes over Boolean variables, each a `Bernoulli` with true as 1. A model uses them
to build a probabilistic circuit, `z ~ AND(x, y)`. They run under
[`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm), and, as for every
deterministic node, their clusters are `out` and the joint over the inputs, so every rule is
belief propagation over `Bernoulli` messages, whatever the factorisation.

| node | function | interfaces |
|---|---|---|
| [`AND`](@ref) | ``\mathrm{out} = \mathrm{in}_1 \wedge \mathrm{in}_2`` | `out`, `in1`, `in2` |
| [`OR`](@ref) | ``\mathrm{out} = \mathrm{in}_1 \vee \mathrm{in}_2`` | `out`, `in1`, `in2` |
| [`NOT`](@ref) | ``\mathrm{out} = \neg \mathrm{in}`` | `out`, `in` |
| [`IMPLY`](@ref) | ``\mathrm{out} = \mathrm{in}_1 \Rightarrow \mathrm{in}_2`` | `out`, `in1`, `in2` |

The message towards `out` has log scale zero; one towards an input is normalised, and declares
the log of its normaliser as its log scale. The joint marginal of two inputs is a `Contingency`
table, rows `in1` and columns `in2`, false first. None of the nodes has an average energy of its
own.

```@setup logic
using MessagePassingRulesBase, StandardMessagePassingRules
```

## Example

```jldoctest logic
julia> using StandardMessagePassingRules, MessagePassingRulesBase, Distributions

julia> message = getresult(@call_message_update_rule(node = OR, target = :out, m = (in1 = Bernoulli(0.5), in2 = Bernoulli(0.5))));

julia> mean(message) ≈ 0.75
true

julia> result = @call_message_update_rule(node = IMPLY, target = :in2, m = (out = Bernoulli(1.0), in1 = Bernoulli(1.0)));

julia> mean(getresult(result)) ≈ 1.0
true
```

## AND

```@example logic
MessagePassingRulesBase.rule_coverage(AND)
```

```@docs
AND
```

## OR

```@example logic
MessagePassingRulesBase.rule_coverage(OR)
```

```@docs
OR
```

## NOT

```@example logic
MessagePassingRulesBase.rule_coverage(NOT)
```

```@docs
NOT
```

## IMPLY

```@example logic
MessagePassingRulesBase.rule_coverage(IMPLY)
```

```@docs
IMPLY
```
