# Derivatives

A model may be differentiated, to fit hyperparameters or to run a gradient-based sampler. Its
derivatives then pass through its [rules](@extref MessagePassingRulesBase glossary-rule).
[`@test_rule_derivatives`](@ref) checks that they do. It builds the rule's inputs from a
parameter `θ`, reduces the rule's output to a number with `summary`, and compares ForwardDiff's
derivative with a central finite difference.

The rule under test is the belief propagation rule towards `out` of the `Gaussian` node of the
[overview](@ref "A node to test"):

```@example derivatives
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
nothing # hide
```

A scalar `θ` sets the mean of the message on `μ` and the variance `v`. A vector `θ` sets the mean
and the variance of the message on `μ`, and the check compares the gradient:

```@example derivatives
@testset "Gaussian: derivatives" begin
    @test_rule_derivatives(
        node = Gaussian, target = :out,
        inputs = θ -> (m = (μ = NormalMeanVariance(θ, 1.0), v = PointMass(θ^2)),),
        at = 1.5, summary = d -> mean(d) + var(d),
    )
    @test_rule_derivatives(
        node = Gaussian, target = :out,
        inputs = θ -> (m = (μ = NormalMeanVariance(θ[1], θ[2]), v = PointMass(0.5)),),
        at = [1.0, 2.0], summary = var,
    )
end
nothing # hide
```

A rule that converts its inputs to `Float64`, or strips the dual part of a number, gives a wrong
or zero derivative and fails. An [in-place rule](@extref MessagePassingRulesBase glossary-in-place-rule)
is checked twice, through `rule` and through `rule!`.

## API

```@docs
@test_rule_derivatives
test_rule_derivatives
```
