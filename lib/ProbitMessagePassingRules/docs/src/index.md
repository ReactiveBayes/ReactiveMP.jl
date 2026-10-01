# ProbitMessagePassingRules

The [`Probit`](@ref) node for reactive message passing: a binary observation explained by a real
latent variable through the standard normal CDF. Use it for probit regression and binary
classification, where each label `y` is linked to a latent score `x`, itself normal, by
`y ~ Probit(x)` in an RxInfer model.

```@docs
ProbitMessagePassingRules
```

!!! info "Where these rules run"
    The [ReactiveMP](https://reactivebayes.github.io/ReactiveMP.jl/dev/) engine runs these rules on
    a factor graph, which [RxInfer](https://github.com/ReactiveBayes/RxInfer.jl) builds from a model.
    The examples here call the rules directly.

The package has a single node, and this page covers it in full.

## Overview

[`Probit`](@ref) observes a binary `out`, or the probability of a `1`, through `Φ(in)`. The
likelihood of `in` under a binary observation is not a normal, so a normal
[message](@extref MessagePassingRulesBase glossary-message) on `in` times that likelihood is not
normal either.

The node's own algorithm, [`ProbitEP`](@ref), keeps the messages normal by
[expectation propagation](@extref MessagePassingRulesBase glossary-expectation-propagation) (EP).
EP computes the mean and variance of the exact product, called the *tilted* distribution, and
returns the normal message that, multiplied by the message on `in`, has that mean and variance.
The message on `in` that the rule reads is called the *cavity*.

The node also runs under
[`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm). There, the message
towards `in` is the exact likelihood as a log-density, for a model that approximates non-normal
messages elsewhere.

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

```@example probit
using ProbitMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase
MessagePassingRulesBase.nodespec(Probit)
```

| name | aliases | meaning | messages and marginals its rules take |
|---|---|---|---|
| `out` | none | the binary output, or the probability of a `1` | `PointMass` or `Bernoulli`, with its value in `[0, 1]` |
| `in` | none | the latent input | a univariate normal, or a `PointMass` towards `out` |

An observed label reaches [`Probit`](@ref) as a
[point mass](@extref MessagePassingRulesBase glossary-point-mass), a distribution with all its
mass on the observed value. The drawn declaration lists the
[initial message](@extref MessagePassingRulesBase glossary-initial-message) on `in`, which the
Algorithm section explains.

## Algorithm

[`ProbitEP`](@ref)`(; p = 32)` is the node's own
[algorithm](@extref MessagePassingRulesBase glossary-algorithm), so a model that names none runs
it. Its keyword `p` is the number of Gauss–Hermite points of the average energy
([`GaussHermiteCubature`](@extref MessagePassingRulesApproximations.GaussHermiteCubature)); the
messages are closed-form and ignore it. The algorithm declares its
[dependencies](@extref MessagePassingRulesBase glossary-dependencies), the inputs each rule
takes:

```@example probit
MessagePassingRulesBase.dependencies_spec(Probit, ProbitEP())
```

The rule towards `in` reads the message on `in` itself, the cavity. Before any message has
arrived there, the node declares `NormalMeanPrecision(0.0, 100.0)` as the cavity's starting value,
which the engine sets wherever the model sets none.

[`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm)`()` must be named by the
model. It declares no dependencies, so its rules follow the
[factorisation](@extref MessagePassingRulesBase glossary-factorisation). They read messages
when `out` and `in` share a [cluster](@extref MessagePassingRulesBase glossary-cluster), and
marginals under [mean field](@extref MessagePassingRulesBase glossary-mean-field). The rule
towards `in` does not read its own edge.

## Supported rules

```@example probit
MessagePassingRulesBase.rule_coverage(Probit)
```

Each cell counts the rules of [`Probit`](@ref) for a target under an algorithm. Under
[`ProbitEP`](@ref) the rules are expectation propagation. The messages read the messages on both
edges. The joint marginal `q(out, in)` exists for a point-mass `out`, and there is an
[average energy](@extref MessagePassingRulesBase glossary-average-energy).

Under `DefaultAlgorithm` there are
[belief propagation](@extref MessagePassingRulesBase glossary-belief-propagation) rules, from
`m(in)` towards `out` and from `m(out)` towards `in`. There are mean-field rules for point-mass
[marginals](@extref MessagePassingRulesBase glossary-marginal), from `q(in)` towards `out` and
from `q(out)` towards `in`, and an average energy. There is no joint marginal rule.

## Example

The expectation-propagation message towards `in`, from an observed `1` and a standard normal
cavity:

```@example probit
@call_message_update_rule(
    node = Probit, target = :in,
    m = (out = PointMass(1.0), in = NormalMeanVariance(0.0, 1.0)),
)
```

The card draws the cavity as an input on the target's own edge. The message times the cavity
has the moments of the tilted distribution, `Φ(in) N(in | 0, 1)` normalised: mean
`√(2 / π) / √2 = 1 / √π ≈ 0.564` and variance `1 - 1 / π ≈ 0.682`:

```@example probit
message = getresult(@call_message_update_rule(
    node = Probit, target = :in,
    m = (out = PointMass(1.0), in = NormalMeanVariance(0.0, 1.0)),
))
mean_var(prod(GenericProd(), NormalMeanVariance(0.0, 1.0), message))
```

The message towards `out`, and the message towards `in` under `DefaultAlgorithm`:

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
- The joint marginal `q(out, in)` exists only under [`ProbitEP`](@ref) and only for a point-mass
  `out`.
- Under `DefaultAlgorithm`, the mean-field rules take point-mass marginals only (`q(in)`
  towards `out`, `q(out)` towards `in`), and the message towards `in` is a
  `ContinuousUnivariateLogPdf`, not a normal: something downstream must approximate it.
- The average energy computes `⟨-log Φ⟩` by Gauss–Hermite cubature: with `ProbitEP(p = …)`
  points, 32 by default, and with 32 under `DefaultAlgorithm`, which has no keyword for it. It
  is exact to rounding while `q(in)` is narrow, a variance up to about 1, and loses accuracy as
  it broadens: about `1e-4` relative at a variance of 25 and `2e-3` at 400, and more points help
  slowly. The messages do not use the cubature.
- No rule declares a [log scale](@extref MessagePassingRulesBase glossary-log-scale): where the
  engine tracks log scales (`logscales = true`), a message from this node carries an
  [`UndefinedLogScale`](@extref MessagePassingRulesBase.UndefinedLogScale).

## API

The node, and the algorithm it runs by default.

```@docs
Probit
ProbitEP
```
