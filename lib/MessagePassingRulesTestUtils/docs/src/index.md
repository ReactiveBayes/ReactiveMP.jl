# MessagePassingRulesTestUtils

A rule package defines [factor nodes](@extref MessagePassingRulesBase glossary-factor-node) and
their [rules](@extref MessagePassingRulesBase glossary-rule) with
[`MessagePassingRulesBase`](@extref MessagePassingRulesBase.MessagePassingRulesBase). This package
tests those rules on their own, the way they are defined, without building a graph. It is a test
dependency only: nothing needs it at run time.

This site is for rule authors, who test the rules of a package of their own. It assumes you have
written a rule, as MessagePassingRulesBase's
[Your first node](@extref MessagePassingRulesBase tutorial-first-node) does.

```@docs
MessagePassingRulesTestUtils
```

## A first test

A node with one rule, the [belief propagation](@extref MessagePassingRulesBase glossary-belief-propagation)
message towards the output of a normal distribution with a known variance, and a table with one
case for it:

```@example index
using MessagePassingRulesBase, MessagePassingRulesTestUtils
using BayesBase, ExponentialFamily, Distributions, Test

struct Gaussian end   # out ~ Normal(μ, v)
Gaussian(μ, v) = NormalMeanVariance(μ, v)

@define_factor_node(node = Gaussian, type = Stochastic, interfaces = [:out, :μ, :v])

@define_message_update_rule(
    node = Gaussian, target = :out,
    args = (m[:μ]::NormalMeanVariance, m[:v]::PointMass),
    logscale = 0,
    body = (args) -> NormalMeanVariance(mean(args.m[:μ]), var(args.m[:μ]) + mean(args.m[:v])),
)

@testset "Gaussian: out" begin
    @test_message_update_rule(
        node = Gaussian, target = :out,
        cases = [(m = (μ = NormalMeanVariance(1.0, 2.0), v = PointMass(0.5)),) => NormalMeanVariance(1.0, 2.5)],
    )
end
nothing # hide
```

The case gives the rule its inbound [messages](@extref MessagePassingRulesBase glossary-message)
and the message it must return. The summary counts eleven checks for the one case: the table
found the rule, compared its result with the expected one, and ran the case again with its inputs
converted to other float types. [Testing your first rule](@ref tutorial-first-rule) explains each
check.

!!! info "Where these rules run"
    The [ReactiveMP](https://reactivebayes.github.io/ReactiveMP.jl/dev/) engine runs these rules
    on a factor graph, which [RxInfer](https://github.com/ReactiveBayes/RxInfer.jl) builds from a
    model. The tests on this site call the rules directly, never inside a model.

## The site

- **Tutorials**: [Testing your first rule](@ref tutorial-first-rule) writes tables for a node's
  belief propagation and variational rules, and reads what a failing case reports.
  [Testing a rule package](@ref tutorial-rule-package) adds verification against the node,
  derivative checks and the rule-coverage gate, as every rule package's suite has them.
- [Table tests](@ref): every check a case makes, tables for marginal rules and average
  energies, and the tolerances.
- [Verification against the node](@ref): a message rule against the node's log-density,
  integrated numerically, and what it can integrate.
- [Derivatives](@ref): automatic differentiation through a rule, against finite differences.
- [The rule-coverage gate](@ref): every rule selected by some test, and how a suite runs the
  gate.
- [Reading a failure](@ref): what each failing check reports, and what to do about it.
