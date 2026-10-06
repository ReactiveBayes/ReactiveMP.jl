# Reading a failure

Every check of this package is a `Test` assertion, reported at the line of the test that made it.
A failing check prints two lines that matter:

- **`Expression`** names the check, such as `rule_output ≈ expected` or `rule_found`.
- **`Evaluated`** says which run failed and why: the case, the rule, the inputs, which inputs were
  converted to another float type, and the values that disagree.

This page shows each kind of failure and what to do about it. It tests one node, a normal
distribution with a known variance:

```@example failures
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
nothing # hide
```

## Showing failures on a page

A failing check inside a `@testset` makes the test set throw when it ends, which would stop this
page. So the page runs its tests in a test set of its own, which records the results without
reporting them. It prints the failures it collected, with the names of types shortened:

```@example failures
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

function failures(ts::Collected; limit = typemax(Int))
    failed = filter(result -> result isa Test.Fail, ts.results)
    for result in first(failed, limit)
        println("Test Failed\n  Expression: ", result.orig_expr, "\n   Evaluated: ", shorten(result.data), "\n")
    end
    length(failed) > limit && println("… and ", length(failed) - limit, " more")
    return nothing
end
nothing # hide
```

In your suite you need none of this. An ordinary `@testset` prints each failure the same way,
with the file and the line of the test, and with each name qualified by its module, such as
`ExponentialFamily.NormalMeanVariance`.

## A wrong value

The expected value of this case is wrong: the variance is `2.5`, not `2.6`.

```@example failures
ts = @testset Collected "Gaussian: out" begin
    @test_message_update_rule(
        node = Gaussian, target = :out,
        cases = [(m = (μ = NormalMeanVariance(1.0, 2.0), v = PointMass(0.5)),) => NormalMeanVariance(1.0, 2.6)],
    )
end
failures(ts; limit = 1)
```

The first report is the case on its own inputs. The others are its runs with the inputs converted
to other float types, and they fail the same way.

**What to do.** When every run of a case fails, the value is wrong, in the rule or in the table.
Work the expected value out again, or check the rule against the node's density with
[`@verify_message_update_rule`](@ref), which needs no expected value.

## A promoted type

When the case on its own inputs passes and only converted runs fail, the result has the wrong
type. This [average energy](@extref MessagePassingRulesBase glossary-average-energy) multiplies by
the constant `2π`, a `Float64`:

```@example failures
@define_average_energy(
    node = Gaussian,
    args = (q[:out]::Any, q[:μ]::Any, q[:v]::PointMass),
    body = (args) -> begin
        y, x, v = args.q[:out], args.q[:μ], mean(args.q[:v])
        (log(2π * v) + (var(y) + var(x) + (mean(y) - mean(x))^2) / v) / 2
    end,
)

ts = @testset Collected "Gaussian: average energy" begin
    @test_average_energy(
        node = Gaussian,
        cases = [(q = (out = NormalMeanVariance(0.0, 1.0), μ = NormalMeanVariance(1.0, 1.0), v = PointMass(1.0)),) => 1.5 + log(2π) / 2],
    )
end
failures(ts)
```

With every input in `Float32`, the energy must be a `Float32`. The product `2π * v` is a
`Float64`, and it makes the whole energy one.

**What to do.** Find the value that does not follow the inputs' type. A `Float64` constant
promotes everything it touches: write `2v * π`, whose `π` takes the type of `2v`, or convert the
constant with `oftype`. A literal such as `0.5` does the same; write `v / 2` instead of `0.5v`.
An explicit `Float64(…)` holds the result at `Float64`, as
[Testing your first rule](@ref tutorial-first-rule) shows.

## No rule for the inputs

The case gives the variance as a number, where the rule takes a `PointMass`:

```@example failures
ts = @testset Collected "Gaussian: out" begin
    @test_message_update_rule(
        node = Gaussian, target = :out,
        cases = [(m = (μ = NormalMeanVariance(1.0, 2.0), v = 0.5),) => NormalMeanVariance(1.0, 2.5)],
    )
end
failures(ts)
```

The check is `rule_found`. The report is the base package's
[`RuleNotFoundError`](@extref MessagePassingRulesBase.RuleNotFoundError): it lists the rules
close to the inputs, with a `✓` for each input that fits and a `✗` for each that does not.

**What to do.** Read the `✗` lines. The case may give a message where the rule takes a marginal,
`m` for `q`, or an input under the wrong name, or of another type. A rule for those inputs may also
be missing, and then the table found a real gap.

## The rule throws

A sign error makes this rule's variance negative, and `sqrt` throws:

```@example failures
@define_message_update_rule(
    node = Gaussian, target = :μ,
    args = (m[:out]::NormalMeanVariance, m[:v]::PointMass),
    logscale = 0,
    body = (args) -> Normal(mean(args.m[:out]), sqrt(var(args.m[:out]) - mean(args.m[:v]))),
)

ts = @testset Collected "Gaussian: μ" begin
    @test_message_update_rule(
        node = Gaussian, target = :μ,
        cases = [(m = (out = NormalMeanVariance(1.0, 0.5), v = PointMass(2.0)),) => Normal(1.0, sqrt(2.5))],
    )
end
failures(ts; limit = 1)
```

The check is `rule_runs`. The report gives the exception and the frames it was thrown from, the
rule's body first. The table goes on with its other cases and runs.

**What to do.** Read the exception and the first frame. Here the variance is a sum, not a
difference.

## A log scale

[`ExpectedWithLogScale`](@ref) checks the [log scale](@extref MessagePassingRulesBase glossary-log-scale)
a rule declares. The rule towards `out` declares `0`, and this case expects `-1`:

```@example failures
ts = @testset Collected "Gaussian: out, log scale" begin
    @test_message_update_rule(
        node = Gaussian, target = :out,
        cases = [(m = (μ = NormalMeanVariance(1.0, 2.0), v = PointMass(0.5)),) => ExpectedWithLogScale(NormalMeanVariance(1.0, 2.5), -1)],
    )
end
failures(ts)
```

The check is `logscale ≈ expected`. A computed log scale must also follow the inputs' float type,
like the result. A log scale of the wrong type fails with a report that names both types.

**What to do.** [`@verify_message_update_rule`](@ref) computes the log scale a belief propagation
message must have, from the node's density, and reports it when it differs from the declared one.

## A verification

Verification reports two checks. `shape_matches_node_definition` fails when the message has the
wrong shape: the log-ratio of the message to the integral of the node's density varies over the
test points. [Testing a rule package](@ref tutorial-rule-package) shows one.
`logscale_matches_node_definition` fails when the shape is right and the declared log scale is
not. This rule declares `1`, where the density implies `0`:

```@example failures
@define_message_update_rule(
    node = Gaussian, target = :out,
    args = (m[:μ]::PointMass, m[:v]::PointMass),
    logscale = 1,
    body = (args) -> NormalMeanVariance(mean(args.m[:μ]), mean(args.m[:v])),
)

ts = @testset Collected "Gaussian: out, verified" begin
    @verify_message_update_rule(node = Gaussian, target = :out, m = (μ = PointMass(1.0), v = PointMass(0.5)))
end
failures(ts)
```

**What to do.** Trust the integral over the hand derivation, after checking that the node's
density, [`nodefunction`](@extref MessagePassingRulesBase.nodefunction), is the one the rule
assumes. [Verification against the node](@ref) lists the inputs verification can integrate.

## A derivative

[`@test_rule_derivatives`](@ref) reports two kinds of failure:

- `automatic_derivative ≈ finite_difference`: the rule runs with ForwardDiff's dual numbers, but
  its derivative differs from a central finite difference. The report gives both derivatives.
- `rule_runs`, labelled with the derivative it was computing: the rule throws on dual numbers.
  The usual cause is a conversion, `Float64(x)`, which has no method for a dual number. The
  report gives the exception and the frames, whose types spell out the dual numbers in full.

**What to do.** Remove the conversion, or write it with the input's own type, `float(x)`. A wrong
derivative from a rule that runs usually means a value computed outside the rule's inputs, such
as a cached one.

## An allocation

With `check_nonallocating = true`, a case also measures the rule's call through the engine's entry
point, which must allocate nothing. This average energy computes the difference of the means as
vectors, the way a rule for vectors would:

```@example failures
@define_average_energy(
    node = Gaussian,
    args = (q[:out]::Any, q[:μ]::Any, q[:v]::PointMass),
    body = (args) -> begin
        y, x, v = args.q[:out], args.q[:μ], mean(args.q[:v])
        d = [mean(y)] .- [mean(x)]
        (log(2v * π) + (var(y) + var(x) + sum(abs2, d)) / v) / 2
    end,
)

ts = @testset Collected "Gaussian: average energy, allocations" begin
    @test_average_energy(
        node = Gaussian,
        cases = [(q = (out = NormalMeanVariance(0.0, 1.0), μ = NormalMeanVariance(1.0, 1.0), v = PointMass(1.0)),) => 1.5 + log(2π) / 2],
        check_nonallocating = true,
    )
end
failures(ts)
```

The check is `allocated_bytes == 0`.

**What to do.** Compute with numbers, `mean(y) - mean(x)`, or with tuples, which do not live on
the heap. A rule for vectors writes into a buffer it is given: an
[in-place rule](@extref MessagePassingRulesBase glossary-in-place-rule) into its output, or a rule
with scratch into its scratch. Julia removes some temporary arrays itself, so a rule may pass
this check on one Julia version and fail it on another.

## Other checks

| Expression | What failed | What to do |
|:-----------|:------------|:-----------|
| `rule!(buffer) === buffer` | An [in-place rule](@extref MessagePassingRulesBase glossary-in-place-rule) returned a new object instead of writing into the buffer from its `preallocate`. | Write into the buffer and return it. |
| `rule! == rule` | The in-place rule and the plain one disagree. | Check that the in-place body overwrites every entry of the buffer. |
| `rule_with_reused_scratch == rule` | A rule with scratch read its scratch before writing it: the result changed after [`poison!`](@ref MessagePassingRulesTestUtils.poison!) filled it with `NaN`. | Write every entry of the scratch before reading it. |
| `annotation == expected` | A rule wrote another annotation than [`ExpectedWithAnnotations`](@ref) expects, or none. | Check the `annotate!` calls of the body. |

The rule-coverage gate fails one check, `isempty(gaps)`, and prints a
[`RuleCoverageGap`](@ref MessagePassingRulesTestUtils.RuleCoverageGap) for each rule no test
selected, with the line that defined the rule. Add a case for that rule, or a verification.
[The rule-coverage gate](@ref) says what counts.
