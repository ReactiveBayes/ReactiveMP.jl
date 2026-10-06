# The rule-coverage gate

A rule package ends its suite with a gate. After every test item has run, each
[rule](@extref MessagePassingRulesBase glossary-rule) the package defines must have been selected
by some test, and each node must have some rule selected. A rule added without a test fails the
suite.

A rule counts as tested only when a test *selected* it, that is, resolved it as the rule to run.
A test selects a rule through:

- a table case, [`@test_message_update_rule`](@ref) and its siblings;
- a verification, [`@verify_message_update_rule`](@ref);
- a derivative check, [`@test_rule_derivatives`](@ref);
- a direct call, [`call_message_update_rule`](@extref MessagePassingRulesBase.call_message_update_rule),
  its siblings or their macros.

## Run the gate

The `Gaussian` node of the [first tutorial](@ref tutorial-first-rule) has three rules here, and no test has
run yet. [`check_rule_coverage`](@ref) takes the modules to check, here the page's own, and
returns a [`RuleCoverageGap`](@ref MessagePassingRulesTestUtils.RuleCoverageGap) for each rule and
node that no test exercised:

```@example coverage
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

@define_message_update_rule(
    node = Gaussian, target = :μ,
    args = (m[:out]::PointMass, m[:v]::PointMass),
    logscale = 0,
    body = (args) -> NormalMeanVariance(mean(args.m[:out]), mean(args.m[:v])),
)

foreach(println, check_rule_coverage(@__MODULE__))
```

Each gap names a rule by its target, its algorithm and the line that defined it. The last gap is
the node's.

A table case selects the belief propagation rule towards `out`, and a direct call selects the
variational one:

```@example coverage
@testset "Gaussian: out" begin
    @test_message_update_rule(
        node = Gaussian, target = :out,
        cases = [(m = (μ = NormalMeanVariance(1.0, 2.0), v = PointMass(0.5)),) => NormalMeanVariance(1.0, 2.5)],
    )
end

vmp = @call_message_update_rule(node = Gaussian, target = :out, q = (μ = NormalMeanVariance(1.0, 2.0), v = PointMass(0.5)))

foreach(println, check_rule_coverage(@__MODULE__))
```

The node has a selected rule, so its gap is gone. The rule towards `μ` is still reported. A
verification selects it, and the gate passes:

```@example coverage
@testset "Gaussian: μ" begin
    @verify_message_update_rule(node = Gaussian, target = :μ, m = (out = PointMass(3.0), v = PointMass(0.5)))
end

check_rule_coverage(@__MODULE__)
```

[`rule_test_locations`](@ref MessagePassingRulesTestUtils.rule_test_locations) says where the tests
that selected a rule were written. A direct call has no source line:

```@example coverage
MessagePassingRulesTestUtils.rule_test_locations(getrule(vmp))
```

## What does not count

A broader rule that would also have answered a case stays reported, so every method of a rule
needs a case of its own. A rule reached only through a graph is not counted, since an engine
resolves rules itself.

## Wire the gate into a suite

Selections accumulate over the whole process, from the moment the package is loaded. The gate
means something only after an unfiltered run. So `test/runtests.jl` notes whether its filter left
any test item out, and checks the gate only when none was. The docstring of
[`check_rule_coverage`](@ref) below shows that file. A suite that tags items `:slow` and skips
them by default runs the gate only with them, `TEST_ALL=true`.

A package extension registers its rules in its own module. Give the gate that module as well:
`check_rule_coverage(MyRules, Base.get_extension(MyRules, :MyRulesExt))`.

## API

```@docs
check_rule_coverage
MessagePassingRulesTestUtils.RuleCoverageGap
MessagePassingRulesTestUtils.rule_test_locations
```
