# [BIFMHelper](@id page-bifm-helper)

## Overview

[`BIFMHelper`](@ref) starts a chain of [`BIFM`](@ref) nodes, between the prior of the first state
and the first state itself. It is where the backward pass turns into the forward one: it passes
the backward [message](@extref MessagePassingRulesBase glossary-message) through towards the
prior, and sends the prior's [marginal](@extref MessagePassingRulesBase glossary-marginal) forward
as the chain's starting marginal.

## Definition

In an RxInfer model the helper is written `out ~ BIFMHelper(in)`, with `out` the first state and
`in` its prior. As a density it is the
identity, `p(out | in) = δ(out - in)`; its rules implement the switch between the passes rather
than a product with it.

## Interfaces

```@example bifm-helper
using BIFMMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily
MessagePassingRulesBase.nodespec(BIFMHelper)
```

The node has two [interfaces](@extref MessagePassingRulesBase glossary-interface).

| name | meaning | messages and marginals the rules take |
|---|---|---|
| `out` | the first state of the chain | the message on `out`, any type, towards `in` |
| `in` | the prior of the first state | the marginal `q(in)`, any type, towards `out` |

Towards `in` the rule passes the message on `out` through unchanged; towards `out` it sends
`TerminalProdArgument(q(in))`, which the first [`BIFM`](@ref) reads as the marginal of its
`zprev`.

## Algorithm

`BIFMHelper` runs under the default [algorithm](@extref MessagePassingRulesBase glossary-algorithm),
[`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm), and a model names none.
The model keeps `in` and `out` in separate
[clusters](@extref MessagePassingRulesBase glossary-cluster), `q(in) q(out)`. The node declares
its [dependencies](@extref MessagePassingRulesBase glossary-dependencies), the inputs each
target's rule takes:

```@example bifm-helper
MessagePassingRulesBase.dependencies_spec(BIFMHelper, DefaultAlgorithm())
```

The rule towards `out` takes the input of the
[default scheme](@extref MessagePassingRulesBase glossary-default-scheme), the marginal of `in`.
The rule towards `in` takes the message on `out` instead, which the default scheme would not
give it under `q(in) q(out)`.

## Supported rules

```@example bifm-helper
MessagePassingRulesBase.rule_coverage(BIFMHelper)
```

The rule towards `in` reads the message on `out` and the rule towards `out` the marginal of
`in`, as the switch between the passes needs. The
[average energy](@extref MessagePassingRulesBase glossary-average-energy) exists only to throw a
[`BIFMFreeEnergyError`](@ref BIFMMessagePassingRules.BIFMFreeEnergyError).

## Example

At the start of the forward pass, the helper sends the marginal of the prior towards the first
state:

```@example bifm-helper
@call_message_update_rule(
    node = BIFMHelper, target = :out,
    q = (in = MvNormalMeanCovariance([1.0], [2.0;;]),),
)
```

[`@call_message_update_rule`](@extref MessagePassingRulesBase.@call_message_update_rule) runs
the rule and draws the marginal it read. The result wraps that marginal in a
`TerminalProdArgument`, which [`getresult`](@extref MessagePassingRulesBase.getresult) returns:

```jldoctest
julia> using BIFMMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily

julia> result = @call_message_update_rule(
           node = BIFMHelper, target = :out,
           q = (in = MvNormalMeanCovariance([1.0], [2.0;;]),),
       );

julia> getresult(result) isa TerminalProdArgument
true
```

At the end of the backward pass, the helper passes the backward message on the first state
through towards the prior, unchanged:

```@example bifm-helper
@call_message_update_rule(
    node = BIFMHelper, target = :in,
    m = (out = MvNormalWeightedMeanPrecision([1.0], [0.5;;]),),
)
```

## Limitations

- **No free energy**: its average energy throws a
  [`BIFMFreeEnergyError`](@ref BIFMMessagePassingRules.BIFMFreeEnergyError).
- It is meaningful only at the start of a chain of [`BIFM`](@ref) nodes.

## API

```@docs
BIFMHelper
```
