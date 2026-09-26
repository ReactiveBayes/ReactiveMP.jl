# ProbitMessagePassingRules

The probit node for reactive message passing: a binary observation explained by a real latent
variable through the standard normal CDF. Use it for probit regression and binary
classification, where each label `y` is linked to a latent score `x ~ N(…)` by
`y ~ Probit(x)`.

```@docs
ProbitMessagePassingRules
```

!!! info "Where these rules run"
    This package defines message passing rules; it does not build or run models. The
    [ReactiveMP](https://reactivebayes.github.io/ReactiveMP.jl/dev/) engine runs the rules on a
    factor graph, and [RxInfer](https://github.com/ReactiveBayes/RxInfer.jl) builds that graph from
    a model written with [GraphPPL](https://github.com/ReactiveBayes/GraphPPL.jl). The examples
    here call the rules directly, as a test or an interactive session does.

The package has a single node, and this page covers it in full.

## Overview

[`Probit`](@ref) observes a binary `out`, or the probability of a `1`, through `Φ(in)`. The
likelihood of `in` under a binary observation is not a normal, so the node's own algorithm,
[`ProbitEP`](@ref), projects it onto one by expectation propagation: the message towards `in`
is the normal that, multiplied by the message on `in`, matches the moments of the exact
posterior of `in`. The node also runs under
[`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm), whose message towards
`in` is the exact likelihood as a log-density, for a model that handles non-normal messages
downstream.

## Definition

With `Φ` the standard normal CDF and `out ∈ {0, 1}`, or `out = p ∈ [0, 1]` for a soft label,

```math
p(\mathrm{out} \mid \mathrm{in}) = \Phi(\mathrm{in})^{\mathrm{out}} \, \bigl(1 - \Phi(\mathrm{in})\bigr)^{1 - \mathrm{out}},
\qquad
\Phi(x) = \int_{-\infty}^{x} \mathcal{N}(t \mid 0, 1) \, \mathrm{d}t.
```

Integrated against a normal message `N(in | μ, v)`, the probability of a `1` is
`Φ(μ / √(1 + v))`, which is the message towards `out`.

## Interfaces

| name | aliases | meaning | messages and marginals its rules take |
|---|---|---|---|
| `out` | | the binary output, or the probability of a `1` | `PointMass` or `Bernoulli`, with its value in `[0, 1]` |
| `in` | | the latent input | a univariate normal, or a `PointMass` towards `out` |

## Algorithm

[`ProbitEP`](@ref)`(; p = 32)` is the node's own algorithm, so a model that names none runs it.
Its keyword `p` is the number of Gauss–Hermite points of the average energy; the messages are
closed-form and ignore it. Its rule towards `in` reads the message on `in` itself, the cavity,
and the node declares `NormalMeanPrecision(0.0, 100.0)` as that message's starting value, which
the engine sets wherever the model sets none.

[`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm)`()` must be named by the
model. Its rules follow the factorisation: they read messages when `out` and `in` share a
cluster and marginals under mean field, and the rule towards `in` does not read its own edge.

## Supported rules

```@example
using MessagePassingRulesBase, ProbitMessagePassingRules # hide
MessagePassingRulesBase.rule_coverage(Probit)
```

Each cell counts the rules for a target under an algorithm. Under `ProbitEP` the rules are
expectation propagation: belief-propagation-style messages from the messages on both edges,
the joint marginal `q(out, in)` for a point-mass `out`, and an average energy. Under
`DefaultAlgorithm` there are belief-propagation rules (from `m(in)` towards `out`, from `m(out)`
towards `in`), mean-field rules for point-mass marginals (`q(in)` towards `out`, `q(out)`
towards `in`), and an average energy; there is no joint marginal rule.

## Example

A message towards `out` from a normal belief about `in`, and the expectation-propagation
message back towards `in` from an observed `1`:

```jldoctest
julia> using ProbitMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase

julia> result = @call_message_update_rule(node = Probit, target = :out, m = (in = NormalMeanVariance(1.0, 0.5),));

julia> mean(getresult(result)) ≈ 0.7928919108787374   # Φ(1 / √(1 + 0.5))
true

julia> result = @call_message_update_rule(
           node = Probit, target = :in, m = (out = PointMass(1.0), in = NormalMeanPrecision(0.0, 1.0)),
       );

julia> getresult(result) isa NormalWeightedMeanPrecision
true

julia> result = @call_message_update_rule(
           node = Probit, target = :in, algorithm = DefaultAlgorithm(), m = (out = PointMass(1.0),),
       );

julia> getresult(result) isa ContinuousUnivariateLogPdf
true
```

## Limitations

- `in` is univariate; there are no multivariate rules.
- The value on `out` must lie in `[0, 1]`; the expectation-propagation rules towards `in` and
  the joint marginal throw an `ArgumentError` otherwise.
- The joint marginal `q(out, in)` exists only under `ProbitEP` and only for a point-mass `out`.
- Under `DefaultAlgorithm`, the mean-field rules take point-mass marginals only (`q(in)`
  towards `out`, `q(out)` towards `in`), and the message towards `in` is a
  `ContinuousUnivariateLogPdf`, not a normal: something downstream must approximate it.
- The average energy under `DefaultAlgorithm` always uses 32 Gauss–Hermite points; only
  `ProbitEP` makes the number configurable.
- No rule declares a log scale: where the engine tracks log scales (`logscales = true`), a
  message from this node carries an `UndefinedLogScale`.

## API

The node, and the algorithm it runs by default.

```@docs
Probit
ProbitEP
```
