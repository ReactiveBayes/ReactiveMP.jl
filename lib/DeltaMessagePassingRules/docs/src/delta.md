# [The Delta node](@id delta-node)

```@meta
DocTestSetup = :(using DeltaMessagePassingRules, MessagePassingRulesBase, MessagePassingRulesApproximations, ExponentialFamily, BayesBase)
```

## Overview

The Delta node puts a deterministic function into a
[factor graph](@extref MessagePassingRulesBase glossary-factor-graph). In a model written
with [RxInfer](https://reactivebayes.github.io/RxInfer.jl/stable/), a line such as `z := f(x, y)` creates a [`DeltaFn`](@ref) node with `z` on its output `out` and `x`, `y` on
its inputs. Use it when `f` has no node with exact rules, for example a nonlinear sensor model
or a change of coordinates.

The node's rules push [messages](@extref MessagePassingRulesBase glossary-message) through `f`
approximately. The model names the approximation method in a [`DeltaApproximation`](@ref). When
the model also gives the inverse of `f`, the messages back towards the inputs go through that
inverse. Without one, they come from the joint belief over the inputs.

## Definition

The node is a [deterministic node](@extref MessagePassingRulesBase glossary-deterministic-node):
its density is a Dirac delta, which puts all its mass on `out = f(in₁, …, inₙ)`,

```math
p(\mathrm{out} \mid \mathrm{in}_1, \ldots, \mathrm{in}_n) = \delta\big(\mathrm{out} - f(\mathrm{in}_1, \ldots, \mathrm{in}_n)\big).
```

The exact message towards `out` is the
[pushforward](@extref MessagePassingRulesBase glossary-pushforward) of the messages on the
inputs: the distribution of `f(in)` when each input follows its message. For a nonlinear `f`
it is not normal, even when the inputs' messages are. The Gaussian methods replace it by the
normal with the same mean and covariance:

```math
\overrightarrow{\mu}(\mathrm{out}) \approx \mathcal{N}\big(\mathrm{out} \mid \mathrm{E}[f(\mathrm{in})], \mathrm{Cov}[f(\mathrm{in})]\big),
\qquad \mathrm{in}_k \sim \overrightarrow{\mu}(\mathrm{in}_k).
```

[`Unscented`](@extref MessagePassingRulesApproximations.Unscented) estimates these moments from
sigma points, and [`Linearization`](@extref MessagePassingRulesApproximations.Linearization)
from the first-order expansion of `f` at the inputs' means. Towards the inputs, the joint belief
over them is the forward joint corrected by the message from `out`, with a Rauch–Tung–Striebel
step, [`smoothRTS`](@extref MessagePassingRulesApproximations.smoothRTS).

## Interfaces

```@example delta
using DeltaMessagePassingRules, MessagePassingRulesBase, MessagePassingRulesApproximations, ExponentialFamily, BayesBase
MessagePassingRulesBase.nodespec(DeltaFn)
```

| name | aliases | meaning | messages and marginals the rules take |
|---|---|---|---|
| `out` | none | the function's value | a normal message `m[:out]` (Gaussian methods); any message and `q(out)` ([`CVIProjection`](@ref)) |
| `in` | none | the group of random inputs, `(:in, k)`; inputs on constants or data are folded into `f` | normal messages `m[:in]` and, without an inverse, the joint `q[(:in,)]` as a `JointNormal` (Gaussian methods); any messages and the joint as a `FactorizedJoint` ([`CVIProjection`](@ref)) |

The input `in` of [`DeltaFn`](@ref) is a [group](@extref MessagePassingRulesBase glossary-group):
an interface with one member per random input. Inputs connected to constants or data are folded
into `f`, which the declaration shows as `static inputs: fold`. The node's [clusters](@extref MessagePassingRulesBase glossary-cluster) are
`out` and the joint over its inputs, as for every deterministic node, so the node takes no
[factorisation](@extref MessagePassingRulesBase glossary-factorisation) constraint.

## Algorithm

[`DeltaApproximation`](@ref) is the node's
[algorithm](@extref MessagePassingRulesBase glossary-algorithm), and a model must name it: the
node's default algorithm,
[`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm), has no rules. It takes
two keywords:

- `method`, required: [`Unscented`](@extref MessagePassingRulesApproximations.Unscented)`()`,
  [`Linearization`](@extref MessagePassingRulesApproximations.Linearization)`()`, or
  [`CVIProjection`](@ref)`()`, once ExponentialFamilyProjection is loaded;
- `inverse`, default `nothing`: a known inverse of `f`, or a tuple of them, one per input.

The two forms of the algorithm declare different
[dependencies](@extref MessagePassingRulesBase glossary-dependencies) towards an input. Without
an inverse, the message towards `(:in, k)` reads that input's own message `m[:in][k]` and the
joint `q[(:in,)]`:

```@example delta
MessagePassingRulesBase.dependencies_spec(DeltaFn, DeltaApproximation(method = Unscented()))
```

With an inverse, it reads the message from `out` and the other inputs' messages `m[:in][!k]`:

```@example delta
MessagePassingRulesBase.dependencies_spec(DeltaFn, DeltaApproximation(method = Unscented(), inverse = y -> y))
```

## Supported rules

```@example delta
MessagePassingRulesBase.rule_coverage(DeltaFn)
```

Each [`DeltaApproximation`](@ref) column of the [`DeltaFn`](@ref) table is one form of the algorithm that rules are written for: the
Gaussian methods in general, then without and with a known inverse. The `q(in)` row is the rule
for the joint over the inputs.

Every rule is a [belief propagation](@extref MessagePassingRulesBase glossary-belief-propagation)
rule on messages, except the rule towards an input without an inverse, which also reads the
joint over the inputs. There are no
[mean-field](@extref MessagePassingRulesBase glossary-mean-field) rules. The table leaves out
the rules of [`CVIProjection`](@ref), since they load with their extension
([Projection](@ref delta-projection)).

The node has no [average energy](@extref MessagePassingRulesBase glossary-average-energy). Its
term in the [Bethe free energy](@extref MessagePassingRulesBase glossary-bethe-free-energy) is
minus the entropy of the joint over its inputs, which the engine computes from that marginal.

## Example

The message towards the input of `out = 2in + 1`, through its known inverse, by linearisation.
The inverse belongs to the algorithm, so the call needs no node object:

```@example delta
algorithm = DeltaApproximation(method = Linearization(), inverse = y -> (y - 1) / 2)

@call_message_update_rule(
    node = DeltaFn, target = (:in, 1), algorithm = algorithm,
    m = (out = NormalMeanVariance(3.0, 2.0), in = (nothing,)),
)
```

The card draws the message from `out` as the input and the message towards `in[1]` as the
target. A normal with mean `3` and variance `2` on `out` maps through `(y - 1) / 2` to mean `1`
and variance `2 / 4 = 0.5`:

```jldoctest delta
julia> algorithm = DeltaApproximation(method = Linearization(), inverse = y -> (y - 1) / 2);

julia> result = @call_message_update_rule(
           node = DeltaFn, target = (:in, 1), algorithm = algorithm,
           m = (out = NormalMeanVariance(3.0, 2.0), in = (nothing,)),
       );

julia> all(mean_var(getresult(result)) .≈ (1.0, 0.5))
true
```

Without the inverse, the same message takes two rules. The rules towards `out` and the joint
read `f` from the node object with
[`getnodefn`](@extref MessagePassingRulesBase.getnodefn). An engine provides that object, and a
small stand-in holds it here, as on the [overview](index.md). First the joint over the inputs,
from the input's message and the message from `out`:

```@example delta
struct WithFunction{F}
    f::F
end

MessagePassingRulesBase.getnodefn(node::WithFunction, ::MessagePassingRulesBase.Target{:out}) = node.f

ctx = MessagePassingRulesBase.RuleContext(node = WithFunction(x -> 2x + 1))
unknown = DeltaApproximation(method = Linearization())

joint = @call_marginal_update_rule(
    node = DeltaFn, target = (:in,), algorithm = unknown, ctx = ctx,
    m = (out = NormalMeanVariance(3.0, 2.0), in = (NormalMeanVariance(1.0, 0.5),)),
)
q_in = getresult(joint)
```

Then the message towards the input divides the joint by the input's own message. An
interactive call passes the joint through `clusters`, keyed by its members:

```@example delta
message = @call_message_update_rule(
    node = DeltaFn, target = (:in, 1), algorithm = unknown,
    m = (in = (NormalMeanVariance(1.0, 0.5),),), clusters = ((:in,) => q_in,),
)
mean_var(getresult(message))
```

It is the message the inverse gave, mean `1` and variance `0.5`, since `f` is affine.

## Limitations

- The node has rules only under a [`DeltaApproximation`](@ref), which has no default method.
- [`Unscented`](@extref MessagePassingRulesApproximations.Unscented) and
  [`Linearization`](@extref MessagePassingRulesApproximations.Linearization) take normal messages
  only, univariate or multivariate, and give normal messages. `Linearization` needs `f`
  differentiable by ForwardDiff.
- Without an inverse, the message towards an input divides the input's share of the joint by the
  input's own message. The difference of the precisions need not be positive definite, and the
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
