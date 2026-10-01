# FlowMessagePassingRules

The [`Flow`](@ref) node of the ReactiveMP ecosystem: a deterministic, invertible transformation
``y = f(x)`` of a multivariate variable, with ``f`` a normalising-flow model built from layers.
Use it to give a Gaussian latent variable a flexible, non-Gaussian image, or to learn a
nonlinear map whose inverse and Jacobian are cheap. The flow models themselves, and their
layers, live in this package too: see [Flow models](@ref flow-models).

```@docs
FlowMessagePassingRules
```

!!! info "Where these rules run"
    [ReactiveMP](https://reactivebayes.github.io/ReactiveMP.jl/dev/) runs these rules on a factor
    graph, which [RxInfer](https://github.com/ReactiveBayes/RxInfer.jl) builds from a
    [GraphPPL](https://github.com/ReactiveBayes/GraphPPL.jl) model. The examples here call the
    rules directly, as a test does.

The site has three pages: this one, the node; [Flow models](@ref flow-models), the models and
their layers; and [Internals](@ref flow-internals), the helpers the rules are built on.

## Overview

`Flow` is a [deterministic node](@extref MessagePassingRulesBase glossary-deterministic-node):
it connects an input `in` and an output `out` of the same dimension through a compiled flow
model. A [message](@extref MessagePassingRulesBase glossary-message) arriving on either side is
pushed through the model, forwards through ``f`` or backwards through ``f^{-1}``. It leaves as a
multivariate normal, by linearisation with the model's own Jacobians or by the unscented
transform. Both directions are exact for an affine flow and approximate otherwise.

## Definition

```math
p(y \mid x) = \delta\big(y - f(x)\big), \qquad x, y \in \mathbb{R}^d,
```

where ``f = f_L \circ \dots \circ f_1`` is the composition of the model's layers, each
invertible with a known Jacobian. The Dirac delta puts all the probability on ``y = f(x)``: the
output is a function of the input, with no noise.

## Interfaces

```@example flow
using FlowMessagePassingRules, MessagePassingRulesBase, MessagePassingRulesApproximations, BayesBase, ExponentialFamily
MessagePassingRulesBase.nodespec(Flow)
```

The node has two [interfaces](@extref MessagePassingRulesBase glossary-interface).

| name | aliases | meaning | messages the rules take |
|---|---|---|---|
| `out` | | the output ``y = f(x)`` | multivariate normal |
| `in` | | the input ``x`` | multivariate normal |

Under `Linearization` the rules take `MvNormalMeanCovariance`, `MvNormalMeanPrecision` and
`MvNormalWeightedMeanPrecision`, and return an `MvNormalMeanCovariance` for the first and an
`MvNormalMeanPrecision` for the others; under `Unscented` they take any
multivariate normal and return an `MvNormalMeanCovariance`. No rule reads a
[marginal](@extref MessagePassingRulesBase glossary-marginal).

## Algorithm

The node's [algorithm](@extref MessagePassingRulesBase glossary-algorithm),
[`FlowApproximation`](@ref)`(model; method = Linearization())`, carries the
[`CompiledFlowModel`](@ref) and the method,
[`Linearization`](@extref MessagePassingRulesApproximations.Linearization)`()` or
[`Unscented`](@extref MessagePassingRulesApproximations.Unscented)`()`. A model must name it for
every `Flow` node: the node has no default algorithm, so under
[`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm) no rule exists. The node
declares no [dependencies](@extref MessagePassingRulesBase glossary-dependencies), so each rule
takes the inputs of the [default scheme](@extref MessagePassingRulesBase glossary-default-scheme):
the message on the other interface.

## Supported rules

```@example flow
MessagePassingRulesBase.rule_coverage(Flow)
```

The two `FlowApproximation` columns are its two methods. The first is linearisation, with three
rules per edge, one per normal form; the second is the unscented transform. Every rule is a
[belief propagation](@extref MessagePassingRulesBase glossary-belief-propagation) message. There
is no marginal rule, since the marginal of `in` is the product of its two messages. There is no
[average energy](@extref MessagePassingRulesBase glossary-average-energy), since a deterministic
node needs none: its contribution to the
[Bethe free energy](@extref MessagePassingRulesBase glossary-bethe-free-energy) is minus the
entropy of `q(in)`, as good as the approximation behind it.

## Example

A two-dimensional model of one coupling layer with a planar flow, given its three parameters, and
the message towards `out` from a standard normal centred at `[1, 2]`:

```@example flow
model = compile(FlowModel(2, (AdditiveCouplingLayer(PlanarFlow(); permute = false),)), [1.0, 2.0, 3.0])

@call_message_update_rule(
    node = Flow, target = :out, algorithm = FlowApproximation(model),
    m = (in = MvNormalMeanCovariance([1.0, 2.0], [1.0 0.0; 0.0 1.0]),),
)
```

[`@call_message_update_rule`](@extref MessagePassingRulesBase.@call_message_update_rule) runs
the rule and draws the message it read. Under linearisation the mean of the result is the image
of the input's mean, [`FlowMessagePassingRules.forward`](@ref) of it. The unscented transform
gives a mean close to it. [`getresult`](@extref MessagePassingRulesBase.getresult) extracts both
for the comparison:

```jldoctest; setup = :(using FlowMessagePassingRules, MessagePassingRulesBase, MessagePassingRulesApproximations, BayesBase, ExponentialFamily)
julia> model = compile(FlowModel(2, (AdditiveCouplingLayer(PlanarFlow(); permute = false),)), [1.0, 2.0, 3.0]);

julia> m_in = MvNormalMeanCovariance([1.0, 2.0], [1.0 0.0; 0.0 1.0]);

julia> linearised = getresult(@call_message_update_rule(node = Flow, target = :out, m = (in = m_in,), algorithm = FlowApproximation(model)));

julia> mean(linearised) ≈ FlowMessagePassingRules.forward(model, [1.0, 2.0])
true

julia> unscented = getresult(@call_message_update_rule(node = Flow, target = :out, m = (in = m_in,), algorithm = FlowApproximation(model; method = Unscented())));

julia> isapprox(mean(unscented), mean(linearised); atol = 1e-2)
true
```

## Limitations

- Multivariate normal messages only, in and out; no rule takes a marginal.
- No average energy, as for every deterministic node, and a free energy only as good as the
  linearisation or the unscented transform behind the messages.
- The model's parameters are fixed once compiled: the node does not learn them.
- Under [`Linearization`](@extref MessagePassingRulesApproximations.Linearization), the rules for a message in precision form evaluate the Jacobian at the
  message's mean read as a point on the other side of the flow, where the covariance form
  evaluates it at the mean itself: the two forms agree for a two-dimensional model of one
  coupling layer, and differ in general for a nonlinear flow.

## API

The node and its algorithm:

```@docs
Flow
FlowApproximation
```
