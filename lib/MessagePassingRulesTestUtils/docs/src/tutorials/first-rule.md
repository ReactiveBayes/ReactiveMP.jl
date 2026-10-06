# [Testing your first rule](@id tutorial-first-rule)

In this tutorial you test the two rules of a node with tables. You write a case for each rule,
learn what a case checks, and read the report of a case that fails.

You need to know what a [rule](@extref MessagePassingRulesBase glossary-rule) is and how one is
defined. MessagePassingRulesBase's [Your first node](@extref MessagePassingRulesBase tutorial-first-node)
derives the rules this page tests.

## The node

The node is a normal distribution with a known variance, ``f(y, x, v) = \mathcal{N}(y \mid x, v)``,
where ``y`` is the output `out`, ``x`` the mean `μ` and ``v`` the variance `v`:

```@example first-rule
using MessagePassingRulesBase, MessagePassingRulesTestUtils
using BayesBase, ExponentialFamily, Distributions, Test

struct Gaussian end   # out ~ Normal(μ, v)
Gaussian(μ, v) = NormalMeanVariance(μ, v)

@define_factor_node(node = Gaussian, type = Stochastic, interfaces = [:out, :μ, :v])
```

The line `Gaussian(μ, v) = NormalMeanVariance(μ, v)` makes the node callable as the distribution
of its output. The [next tutorial](@ref tutorial-rule-package) uses it to check the rules against
the node's density.

## A belief propagation rule

The [belief propagation](@extref MessagePassingRulesBase glossary-belief-propagation) message
towards `out` integrates the factor against the message on `μ`. With a normal message of mean
``m`` and variance ``w`` on `μ`, and a known variance ``v``, it is a normal of mean ``m`` and
variance ``w + v``:

```@example first-rule
@define_message_update_rule(
    node = Gaussian, target = :out,
    args = (m[:μ]::NormalMeanVariance, m[:v]::PointMass),
    logscale = 0,
    body = (args) -> NormalMeanVariance(mean(args.m[:μ]), var(args.m[:μ]) + mean(args.m[:v])),
)
nothing # hide
```

The integral is a normalised density, so the rule's
[log scale](@extref MessagePassingRulesBase glossary-log-scale) is `0`.

## A table

A table lists cases. A case is the inputs of one call to the rule and the result the rule must
return:

```@example first-rule
@testset "Gaussian: out" begin
    @test_message_update_rule(
        node = Gaussian, target = :out,
        cases = [
            (m = (μ = NormalMeanVariance(1.0, 2.0), v = PointMass(0.5)),) => NormalMeanVariance(1.0, 2.5),
        ],
    )
end
nothing # hide
```

The inputs are a `NamedTuple`. Its entry `m` holds the inbound
[messages](@extref MessagePassingRulesBase glossary-message), keyed by the interfaces' names. The
table selects the rule from the inputs' types, as an engine would.

The summary counts eleven checks for the one case:

- **The rule is found.** One rule takes these inputs.
- **The result is the expected one.** The types must match exactly, a
  `NormalMeanVariance{Float64}` for a `NormalMeanVariance{Float64}`. The values must agree within
  the tolerances. [`approximately_equal`](@ref MessagePassingRulesTestUtils.approximately_equal)
  makes the comparison.
- **The result has the promoted type.** The case runs nine more times, with its inputs converted
  to `Float32`, `Float64` and `BigFloat`: both inputs at once, then each alone. With both inputs
  in `Float32`, the result must be a `NormalMeanVariance{Float32}`.

## The log scale

A plain expected value leaves the log scale unchecked. Wrap it in [`ExpectedWithLogScale`](@ref)
to check the log scale the rule declares:

```@example first-rule
@testset "Gaussian: out, with its log scale" begin
    @test_message_update_rule(
        node = Gaussian, target = :out,
        cases = [
            (m = (μ = NormalMeanVariance(1.0, 2.0), v = PointMass(0.5)),) => ExpectedWithLogScale(NormalMeanVariance(1.0, 2.5), 0),
        ],
    )
end
nothing # hide
```

The case makes one more check, the log scale's.

## A variational rule

Under [variational message passing](@extref MessagePassingRulesBase glossary-vmp), the rule
towards `out` reads the [marginal](@extref MessagePassingRulesBase glossary-marginal) of `μ`, not
its message. The message is a normal of the marginal's mean and the variance ``v``:

```@example first-rule
@define_message_update_rule(
    node = Gaussian, target = :out,
    args = (q[:μ]::Any, q[:v]::PointMass),
    body = (args) -> NormalMeanVariance(mean(args.q[:μ]), mean(args.q[:v])),
)
nothing # hide
```

A case for it gives marginals, in the entry `q`. One table holds the cases of both rules:

```@example first-rule
@testset "Gaussian: out, both rules" begin
    @test_message_update_rule(
        node = Gaussian, target = :out,
        cases = [
            (m = (μ = NormalMeanVariance(1.0, 2.0), v = PointMass(0.5)),) => ExpectedWithLogScale(NormalMeanVariance(1.0, 2.5), 0),
            (q = (μ = NormalMeanVariance(1.0, 2.0), v = PointMass(0.5)),) => NormalMeanVariance(1.0, 0.5),
        ],
    )
end
nothing # hide
```

The first case selects the belief propagation rule and the second the variational one, by the
inputs each gives.

## A failing case

```@setup first-rule
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

A rule can be right for `Float64` inputs and still fail its table. This version of the
variational rule converts the variance to `Float64`:

```@example first-rule
@define_message_update_rule(
    node = Gaussian, target = :out,
    args = (q[:μ]::Any, q[:v]::PointMass),
    body = (args) -> NormalMeanVariance(mean(args.q[:μ]), Float64(mean(args.q[:v]))),
)
nothing # hide
```

Run its case again. The page collects the failures and prints them, as
[Reading a failure](@ref) shows. In your suite, each prints the same way, at the line of the
table:

```@example first-rule
ts = @testset Collected "Gaussian: out, variational" begin
    @test_message_update_rule(
        node = Gaussian, target = :out,
        cases = [(q = (μ = NormalMeanVariance(1.0, 2.0), v = PointMass(0.5)),) => NormalMeanVariance(1.0, 0.5)],
    )
end
failures(ts)
```

Read a report from the top:

- **`Expression`** names the check: `rule_output ≈ expected` compares the result with the expected
  one.
- **`Evaluated`** says which run failed: the case, the rule, the inputs, and which inputs were
  converted to which float type. Then it gives both values, each with its type.

The value is right in both reports, and the type is wrong. Both runs convert `v`: with every input
in `Float32`, and with `v` alone in `BigFloat`. The result must follow `v`'s type, and the rule's
`Float64(…)` holds it at `Float64`. Remove
the conversion, and the case passes:

```@example first-rule
@define_message_update_rule(
    node = Gaussian, target = :out,
    args = (q[:μ]::Any, q[:v]::PointMass),
    body = (args) -> NormalMeanVariance(mean(args.q[:μ]), mean(args.q[:v])),
)

@testset "Gaussian: out, variational" begin
    @test_message_update_rule(
        node = Gaussian, target = :out,
        cases = [(q = (μ = NormalMeanVariance(1.0, 2.0), v = PointMass(0.5)),) => NormalMeanVariance(1.0, 0.5)],
    )
end
nothing # hide
```

## Next

The expected values in these tables are worked out by hand, so they can repeat a mistake of the
rule's derivation. [Testing a rule package](@ref tutorial-rule-package) checks the rules against
the node's density instead, adds derivative checks, and makes the suite fail when a rule has no
test. [Table tests](@ref) lists every keyword of a table.
