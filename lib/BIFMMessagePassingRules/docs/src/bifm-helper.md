# [BIFMHelper](@id page-bifm-helper)

## Overview

`BIFMHelper` starts a chain of [`BIFM`](@ref) nodes, between the prior of the first state and the
first state itself. It is where the backward pass turns into the forward one: it passes the
backward message through towards the prior, and sends the prior's marginal forward as the
chain's starting marginal.

## Definition

`out ~ BIFMHelper(in)`, with `out` the first state and `in` its prior. As a density it is the
identity, `p(out | in) = δ(out - in)`; its rules implement the switch between the passes rather
than a product with it.

## Interfaces

| name | meaning | messages and marginals the rules take |
|---|---|---|
| `out` | the first state of the chain | the message on `out`, any type, towards `in` |
| `in` | the prior of the first state | the marginal `q(in)`, any type, towards `out` |

Towards `in` the rule passes the message on `out` through unchanged; towards `out` it sends
`TerminalProdArgument(q(in))`, which the first [`BIFM`](@ref) reads as the marginal of its
`zprev`.

## Algorithm

`BIFMHelper` runs under the default algorithm,
[`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm), and a model names none.
The model keeps `in` and `out` in separate clusters, `q(in) q(out)`.

## Supported rules

```@example
using MessagePassingRulesBase, BIFMMessagePassingRules # hide
MessagePassingRulesBase.rule_coverage(BIFMHelper)
```

The rule towards `in` reads the message on `out` and the rule towards `out` the marginal of
`in`, as the switch between the passes needs. The average energy exists only to throw a
[`BIFMFreeEnergyError`](@ref BIFMMessagePassingRules.BIFMFreeEnergyError).

## Example

```jldoctest
julia> using BIFMMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily

julia> result = @call_message_update_rule(
           node = BIFMHelper, target = :out,
           q = (in = MvNormalMeanCovariance([1.0], [2.0;;]),),
       );

julia> getresult(result) isa TerminalProdArgument
true
```

In a model, at the start of the chain:

```julia
z_prior ~ MvNormalMeanPrecision(zeros(2), diageye(2))
z[1] ~ BIFMHelper(z_prior)
```

## Limitations

- **No free energy**: its average energy throws a
  [`BIFMFreeEnergyError`](@ref BIFMMessagePassingRules.BIFMFreeEnergyError).
- It is meaningful only at the start of a chain of [`BIFM`](@ref) nodes.

## API

```@docs
BIFMHelper
```
