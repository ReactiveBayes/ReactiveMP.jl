# Logic

The logic nodes compute Boolean functions of Boolean variables. A model uses them to build a
probabilistic circuit, such as `z ~ AND(x, y)` in RxInfer's syntax. Each variable carries a `Bernoulli` belief, the
probability that it is true.

The nodes are [deterministic](@extref MessagePassingRulesBase glossary-deterministic-node): their
[clusters](@extref MessagePassingRulesBase glossary-cluster) are `out` and the joint over the
inputs, whatever the [factorisation](@extref MessagePassingRulesBase glossary-factorisation).
Every rule is therefore exact
[belief propagation](@extref MessagePassingRulesBase glossary-belief-propagation) over
`Bernoulli` [messages](@extref MessagePassingRulesBase glossary-message). The message towards
`out` is the probability that the function is true, given independent inputs. The message towards
an input weighs each of its two values by the message on `out` at the function's output,
averaged over the message on the other input. The nodes run under
[`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm).

| node | function | interfaces |
|---|---|---|
| [`AND`](@ref) | ``\mathrm{out} = \mathrm{in}_1 \wedge \mathrm{in}_2`` | `out`, `in1`, `in2` |
| [`OR`](@ref) | ``\mathrm{out} = \mathrm{in}_1 \vee \mathrm{in}_2`` | `out`, `in1`, `in2` |
| [`NOT`](@ref) | ``\mathrm{out} = \neg \mathrm{in}`` | `out`, `in` |
| [`IMPLY`](@ref) | ``\mathrm{out} = \mathrm{in}_1 \Rightarrow \mathrm{in}_2`` | `out`, `in1`, `in2` |

The message towards `out` has [log scale](@extref MessagePassingRulesBase glossary-log-scale)
zero. A message towards an input is normalised, and it declares the log of its normaliser as its
log scale. The joint [marginal](@extref MessagePassingRulesBase glossary-marginal) of two inputs
is a `Contingency` table, with rows for `in1` and columns for `in2`, false first. None of the
nodes has an [average energy](@extref MessagePassingRulesBase glossary-average-energy) of its
own.

```@setup logic
using MessagePassingRulesBase, StandardMessagePassingRules, Distributions
```

## Example

Two fair coins, combined by `OR`, are true with probability `3/4`. An implication that is true,
with a true premise, forces its conclusion to be true:

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
MessagePassingRulesBase.nodespec(AND)
```

```@example logic
MessagePassingRulesBase.rule_coverage(AND)
```

The message towards `out` multiplies the probabilities of the inputs:

```@example logic
@call_message_update_rule(node = AND, target = :out, m = (in1 = Bernoulli(0.5), in2 = Bernoulli(0.8)))
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
