# Verification against the node

A table's expected values are derived by hand. Verification checks a message rule against what
it is derived from, the node's density. The [message](@extref MessagePassingRulesBase glossary-message)
a rule computes must be, up to a constant, the node's factor integrated against its other
inputs. Verification computes that integral numerically at a set of points.

With messages as inputs, the reference is [belief propagation](@extref MessagePassingRulesBase glossary-belief-propagation):

```math
\log \mu(x) = \log \int f(x, y_1, \dots, y_n) \prod_j m_j(y_j) \, dy + \text{const}.
```

The constant is the rule's [log scale](@extref MessagePassingRulesBase glossary-log-scale): the
message times the exponential of its log scale is the integral itself. With
[marginals](@extref MessagePassingRulesBase glossary-marginal), the reference is naive
[variational message passing](@extref MessagePassingRulesBase glossary-vmp),
``\log \mu(x) = \mathrm{E}_{q}[\log f(x, y)] + \text{const}``.

## Verify a rule

The node is the `Gaussian` of the [overview](@ref "A node to test"),
``f(y, x, v) = \mathcal{N}(y \mid x, v)``. Verification reads its log-density from
[`nodefunction`](@extref MessagePassingRulesBase.nodefunction), which the declaration defines
when the node is callable as the distribution of its output: here `Gaussian(μ, v)` is a
`NormalMeanVariance`.

```@example verification
using MessagePassingRulesBase, MessagePassingRulesTestUtils
using BayesBase, ExponentialFamily, Distributions, Test

struct Gaussian end
Gaussian(μ, v) = NormalMeanVariance(μ, v)

@define_factor_node(node = Gaussian, type = Stochastic, interfaces = [:out, :μ, :v])

@define_message_update_rule(
    node = Gaussian, target = :out,
    args = (m[:μ]::NormalMeanVariance, m[:v]::PointMass),
    logscale = 0,
    body = (args) -> NormalMeanVariance(mean(args.m[:μ]), var(args.m[:μ]) + mean(args.m[:v])),
)

@define_message_update_rule(
    node = Gaussian, target = :out,
    args = (q[:μ]::Any, q[:v]::PointMass),
    body = (args) -> NormalMeanVariance(mean(args.q[:μ]), mean(args.q[:v])),
)

MessagePassingRulesBase.nodefunction(Gaussian)(out = 1.0, μ = 0.0, v = 2.0)
```

The last line evaluates the log-density at ``y = 1``, ``x = 0`` and ``v = 2``.

Verify the belief propagation rule towards `out`, with a normal message on `μ`, which
verification integrates:

```@example verification
@verify_message_update_rule(node = Gaussian, target = :out, m = (μ = NormalMeanVariance(0.5, 1.5), v = PointMass(2.0)))
```

The result is the log-ratio of the message to the reference at nine quantiles of the message.
It varies by less than `atol`, `1e-6` by default, so the shape is right. It is zero, which is
minus the declared log scale, so the scale is right too.

Verify the variational rule towards `out`, with marginals:

```@example verification
@verify_message_update_rule(node = Gaussian, target = :out, q = (μ = NormalMeanVariance(0.5, 1.5), v = PointMass(2.0)))
```

The log-ratio is again constant, so the shape is right. It is not zero, and nothing checks it:
a variational message has no log scale, and this rule declares none.

Each check is a `Test` assertion. Inside a `@testset`, the summary counts them:

```@example verification
@testset "Gaussian: verification" begin
    @verify_message_update_rule(node = Gaussian, target = :out, m = (μ = NormalMeanVariance(0.5, 1.5), v = PointMass(2.0)))
    @verify_message_update_rule(node = Gaussian, target = :out, q = (μ = NormalMeanVariance(0.5, 1.5), v = PointMass(2.0)))
end
nothing # hide
```

## Limitations

- The node must be [stochastic](@extref MessagePassingRulesBase glossary-stochastic-node) and
  without [groups](@extref MessagePassingRulesBase glossary-group): the reference is its
  [`nodefunction`](@extref MessagePassingRulesBase.nodefunction).
- The inputs are point masses, which are substituted, finite discrete distributions, which are
  enumerated, and at most two continuous univariate distributions, which are integrated.
- The message is univariate. It is compared at its quantiles, or over its support when it is
  discrete.
- A case takes messages or marginals, not both. A rule that mixes them is checked by its table
  only.

## Verify any implementation

[`verify_message_update`](@ref) runs the same check on any function that returns a message and
its log scale, for an implementation that is not a rule of the base package. It takes the
node's log-density, its interface names and the target:

```@example verification
message(m, q) = (NormalMeanVariance(mean(m.μ), var(m.μ) + mean(m.v)), 0.0)
logdensity(; out, μ, v) = logpdf(NormalMeanVariance(μ, v), out)

verify_message_update(message, logdensity, (:out, :μ, :v), :out; m = (μ = NormalMeanVariance(0.5, 1.5), v = PointMass(2.0)))
```

## API

```@docs
@verify_message_update_rule
verify_message_update_rule
verify_message_update
```
