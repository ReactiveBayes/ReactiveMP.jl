# [The Delta node](@id delta-node)

```@meta
DocTestSetup = :(using DeltaMessagePassingRules, MessagePassingRulesBase, MessagePassingRulesApproximations, ExponentialFamily, BayesBase)
```

## Overview

The Delta node puts a deterministic function into a factor graph: `z := f(x, y)` in a model
creates a [`DeltaFn`](@ref) node with `z` on `out` and `x`, `y` on its inputs. Its rules push
messages through `f` approximately, by a method the model names in a
[`DeltaApproximation`](@ref). With a known inverse of `f`, the messages back towards the inputs
go through that inverse; without one, they come from the joint over the inputs.

## Definition

```math
p(\mathrm{out} \mid \mathrm{in}_1, \ldots, \mathrm{in}_n) = \delta\big(\mathrm{out} - f(\mathrm{in}_1, \ldots, \mathrm{in}_n)\big)
```

The Gaussian methods approximate the pushforward of normal inputs by a normal:

```math
\overrightarrow{\mu}(\mathrm{out}) \approx \mathcal{N}\big(\mathrm{out} \mid \mathrm{E}[f(\mathrm{in})], \mathrm{Cov}[f(\mathrm{in})]\big),
\qquad \mathrm{in}_k \sim \overrightarrow{\mu}(\mathrm{in}_k),
```

with the moments by the unscented transform or by the first-order expansion of `f` at the
inputs' means. The joint over the inputs is the forward joint corrected by the message from `out`
with a Rauch–Tung–Striebel step.

## Interfaces

| name | aliases | meaning | messages and marginals the rules take |
|---|---|---|---|
| `out` | none | the function's value | a normal message `m[:out]` (Gaussian methods); any message and `q(out)` ([`CVIProjection`](@ref)) |
| `in` | none | the group of random inputs, `(:in, k)`; inputs on constants or data are folded into `f` | normal messages `m[:in]` and, without an inverse, the joint `q[(:in,)]` as a `JointNormal` (Gaussian methods); any messages and the joint as a `FactorizedJoint` ([`CVIProjection`](@ref)) |

A deterministic node's clusters are `out` and the joint over its inputs, so the node takes no
factorisation constraint.

## Algorithm

```julia
DeltaApproximation(; method, inverse = nothing)
```

A model must name it: the node's default algorithm,
[`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm), has no rules.

- `method`, required: [`Unscented`](@extref MessagePassingRulesApproximations.Unscented)`()`,
  [`Linearization`](@extref MessagePassingRulesApproximations.Linearization)`()`, or
  [`CVIProjection`](@ref)`()`, once ExponentialFamilyProjection is loaded;
- `inverse`, default `nothing`: a known inverse of `f`, or a tuple of them, one per input.

The two forms of the algorithm declare different dependencies towards an input. Without an
inverse, the message towards `(:in, k)` reads `m[:in][k]` and the joint `q[(:in,)]`; with one, it
reads `m[:out]` and the other inputs' messages `m[:in][!k]`.

## Supported rules

```@example
using MessagePassingRulesBase, DeltaMessagePassingRules # hide
MessagePassingRulesBase.rule_coverage(DeltaFn)
```

Each `DeltaApproximation` column is one form of the algorithm the rules are written for: the
Gaussian methods, and those without and with a known inverse; the `q(in)` row is the rule for the
joint over the inputs. The rules run on messages, as in belief propagation, and towards an input
without an inverse on the joint over the inputs too; there are no mean-field rules. The rules of
[`CVIProjection`](@ref) are not in the table, since they load with their extension. The node has
no average energy: its term in the Bethe free energy is minus the entropy of the joint over its
inputs, which the engine computes from that marginal.

## Example

The message towards the input of `out = 2in + 1` through its known inverse, by linearization.
The inverse is the algorithm's, so the call needs no node object:

```jldoctest delta
julia> algorithm = DeltaApproximation(method = Linearization(), inverse = y -> (y - 1) / 2);

julia> result = @call_message_update_rule(
           node = DeltaFn, target = (:in, 1), algorithm = algorithm,
           m = (out = NormalMeanVariance(3.0, 2.0), in = (nothing,)),
       );

julia> all(mean_var(getresult(result)) .≈ (1.0, 0.5))
true
```

## Limitations

- The node has rules only under a [`DeltaApproximation`](@ref), which has no default method.
- [`Unscented`](@extref MessagePassingRulesApproximations.Unscented) and
  [`Linearization`](@extref MessagePassingRulesApproximations.Linearization) take normal messages
  only, univariate or multivariate, and give normal messages. `Linearization` needs `f`
  differentiable by ForwardDiff.
- Without an inverse, the message towards an input divides the input's share of the joint by the
  input's own message; the difference of the precisions need not be positive definite, and the
  result is then not a proper normal.
- A single inverse function serves every input; a node with several inputs usually needs a tuple,
  one per input.
- No average energy; the node contributes to the free energy through the entropy of the joint
  over its inputs.

## API

The node and its algorithm:

```@docs
DeltaFn
DeltaApproximation
```

A package that provides a new approximation method opts it in here:

```@docs
is_delta_node_compatible
```
