# [Testing a rule package](@id tutorial-rule-package)

In this tutorial you test a node's rules the way a rule package's suite does. You check a rule
against the node's density, check its derivatives, and end the suite with a gate that fails when
a rule has no test.

It continues [Testing your first rule](@ref tutorial-first-rule), with the same node.

## What a suite holds

A rule package's suite has four parts:

1. **A table for every rule**, as in the first tutorial: the result each case must return.
2. **Verification against the node**: a message rule's result against the node's density,
   integrated numerically.
3. **Derivative checks**, for rules a model differentiates through.
4. **The rule-coverage gate**, run after everything else: every rule the package defines must have
   been selected by some test.

The tests live in `test/`, one file of test items per node, and `test/runtests.jl` runs them and
then the gate:

```text
MyRules/
├── src/MyRules.jl
└── test/
    ├── runtests.jl          the test items, then the gate
    └── gaussian_tests.jl    the Gaussian node's test items
```

Each check is a `Test` assertion, so the tools go inside a `@testitem` or a `@testset`. This page
uses `@testset`.

## The node

The node is the first tutorial's, a normal distribution with a known variance:

```@example rule-package
using MessagePassingRulesBase, MessagePassingRulesTestUtils
using BayesBase, ExponentialFamily, Distributions, Test

struct Gaussian end   # out ~ Normal(μ, v)
Gaussian(μ, v) = NormalMeanVariance(μ, v)

@define_factor_node(node = Gaussian, type = Stochastic, interfaces = [:out, :μ, :v])
nothing # hide
```

```@setup rule-package
struct Collected <: Test.AbstractTestSet
    description::String
    results::Vector{Any}
end
Collected(description; kwargs...) = Collected(description, Any[])
Test.record(ts::Collected, result) = (push!(ts.results, result); result)
Test.finish(ts::Collected) = ts

# The terminal's colours removed, and module-qualified names, `ExponentialFamily.NormalMeanVariance`,
# shortened to the type's name.
shorten(text) = replace(replace(text, r"\e\[[0-9;]*m" => ""), r"(?:var\"[^\"]*\"\.|\b[A-Z]\w*\.|\b__\w+\.)+(?=[A-Za-z_])" => "")

function failures(ts::Collected)
    for result in ts.results
        result isa Test.Fail || continue
        println("Test Failed\n  Expression: ", result.orig_expr, "\n   Evaluated: ", shorten(result.data), "\n")
    end
end
```

## Check a rule against the node

A table's expected value is worked out by hand, from the same derivation as the rule. A mistake
in the derivation then appears in both, and the table passes. Here the rule towards `out` forgets
the variance of the message on `μ`, and so does its table:

```@example rule-package
@define_message_update_rule(
    node = Gaussian, target = :out,
    args = (m[:μ]::NormalMeanVariance, m[:v]::PointMass),
    logscale = 0,
    body = (args) -> NormalMeanVariance(mean(args.m[:μ]), mean(args.m[:v])),
)

@testset "Gaussian: out" begin
    @test_message_update_rule(
        node = Gaussian, target = :out,
        cases = [(m = (μ = NormalMeanVariance(1.0, 2.0), v = PointMass(0.5)),) => NormalMeanVariance(1.0, 0.5)],
    )
end
nothing # hide
```

[`@verify_message_update_rule`](@ref) does not take an expected value. It integrates the node's
log-density, [`nodefunction`](@extref MessagePassingRulesBase.nodefunction), against the inputs,
and compares the rule's message with that integral at a set of points. The failures are collected
and printed, as in the first tutorial:

```@example rule-package
ts = @testset Collected "Gaussian: out, verified" begin
    @verify_message_update_rule(node = Gaussian, target = :out, m = (μ = NormalMeanVariance(1.0, 2.0), v = PointMass(0.5)))
end
failures(ts)
```

The log-ratio of the message to the integral must be the same at every point, since a message is
defined up to a constant. Here it varies, so the message has the wrong shape. The log scale is not
checked, since it means nothing for a message of the wrong shape.

With the variance of `μ` added, the rule passes both checks: the shape, and the log scale `0` it
declares:

```@example rule-package
@define_message_update_rule(
    node = Gaussian, target = :out,
    args = (m[:μ]::NormalMeanVariance, m[:v]::PointMass),
    logscale = 0,
    body = (args) -> NormalMeanVariance(mean(args.m[:μ]), var(args.m[:μ]) + mean(args.m[:v])),
)

@testset "Gaussian: out, verified" begin
    @verify_message_update_rule(node = Gaussian, target = :out, m = (μ = NormalMeanVariance(1.0, 2.0), v = PointMass(0.5)))
end
nothing # hide
```

Verification needs a [stochastic node](@extref MessagePassingRulesBase glossary-stochastic-node)
whose density it can integrate. [Verification against the node](@ref) lists the inputs it
handles.

## Check derivatives

A model may be differentiated, to fit a hyperparameter or to run a gradient-based sampler. The
derivatives then pass through its rules. [`@test_rule_derivatives`](@ref) builds the rule's inputs
from a parameter `θ`, reduces its result to a number with `summary`, and compares ForwardDiff's
derivative with a finite difference:

```@example rule-package
@testset "Gaussian: out, derivatives" begin
    @test_rule_derivatives(
        node = Gaussian, target = :out,
        inputs = θ -> (m = (μ = NormalMeanVariance(θ, 1.0), v = PointMass(θ^2)),),
        at = 1.5, summary = d -> mean(d) + var(d),
    )
end
nothing # hide
```

A rule that converts its inputs to `Float64` drops ForwardDiff's dual numbers and fails here.
[Derivatives](@ref) says more.

## Gate the suite on coverage

The node gains two more rules, the variational message towards `out` and the belief propagation
message towards `μ` from an observed output:

```@example rule-package
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
nothing # hide
```

No test selects them yet. [`check_rule_coverage`](@ref) lists every rule of the given modules
that no test selected. Here the module is the page's own:

```@example rule-package
foreach(println, check_rule_coverage(@__MODULE__))
```

A rule counts once a test resolves it as the rule to run. A table case does, and so does a
verification:

```@example rule-package
@testset "Gaussian: the other rules" begin
    @test_message_update_rule(
        node = Gaussian, target = :out,
        cases = [(q = (μ = NormalMeanVariance(1.0, 2.0), v = PointMass(0.5)),) => NormalMeanVariance(1.0, 0.5)],
    )
    @verify_message_update_rule(node = Gaussian, target = :μ, m = (out = PointMass(3.0), v = PointMass(0.5)))
end

check_rule_coverage(@__MODULE__)
```

The list is empty, so every rule has a test.

## Run the gate in a suite

Selections add up over the whole process, so the gate means something only after the whole suite
has run. `test/runtests.jl` therefore runs it last, and only when no test item was filtered out.
The docstring of [`check_rule_coverage`](@ref) shows that file.

When the gate fails, it prints each gap and fails one check. A new rule without a test then fails
the package's suite, and its report names the rule and the line that defined it.
[The rule-coverage gate](@ref) says what counts as a test, and what does not.
