# Table tests

A table lists cases. Each case is the inputs of one call to a [rule](@extref MessagePassingRulesBase glossary-rule)
and the result the rule must return. A table is the main test of every rule: it is cheap to
write, it takes one line per case, and it checks each case in several ways at once.

This page tests the `Gaussian` node of the [overview](@ref "A node to test"), a normal
distribution with a known variance, with a marginal rule and an
[average energy](@extref MessagePassingRulesBase glossary-average-energy) added:

```@example tables
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

# The joint of `out` and `μ` under q(y, x) q(v), in the information form.
@define_marginal_update_rule(
    node = Gaussian, target = (:out, :μ),
    args = (m[:out]::NormalMeanVariance, m[:μ]::NormalMeanVariance, q[:v]::PointMass),
    body = (args) -> begin
        (my, vy), (mx, vx), v = mean_var(args.m[:out]), mean_var(args.m[:μ]), mean(args.q[:v])
        MvNormalWeightedMeanPrecision([my / vy, mx / vx], [1 / vy + 1 / v -1 / v; -1 / v 1 / vx + 1 / v])
    end,
)

@define_average_energy(
    node = Gaussian,
    args = (q[:out]::Any, q[:μ]::Any, q[:v]::PointMass),
    body = (args) -> begin
        y, x, v = args.q[:out], args.q[:μ], mean(args.q[:v])
        (log(2v * π) + (var(y) + var(x) + (mean(y) - mean(x))^2) / v) / 2
    end,
)
nothing # hide
```

A table for the message rules towards `out`:

```@example tables
@testset "Gaussian: out" begin
    @test_message_update_rule(
        node = Gaussian, target = :out,
        cases = [
            (m = (μ = NormalMeanVariance(1.0, 2.0), v = PointMass(3.0)),) => NormalMeanVariance(1.0, 5.0),
            (q = (μ = PointMass(1.0), v = PointMass(2.0)),) => NormalMeanVariance(1.0, 2.0),
        ],
    )
end
nothing # hide
```

The first case gives messages and selects the [belief propagation](@extref MessagePassingRulesBase glossary-belief-propagation)
rule. The second gives marginals and selects the [variational](@extref MessagePassingRulesBase glossary-vmp)
one. Each case makes several checks, which the summary counts.

## Cases

A case is `inputs => expected`. The inputs are a `NamedTuple` with any of these entries:

- `m`, the inbound [messages](@extref MessagePassingRulesBase glossary-message), keyed by the
  interfaces' declared names.
- `q`, the [marginals](@extref MessagePassingRulesBase glossary-marginal) of single interfaces,
  keyed the same way. A [group](@extref MessagePassingRulesBase glossary-group) is a tuple of its
  members in order, with `nothing` where the rule does not take a member:
  `m = (T = (PointMass(2.0), nothing),)`.
- `clusters`, the joint marginals of [clusters](@extref MessagePassingRulesBase glossary-cluster),
  as pairs from the members to the joint: `clusters = ((:out, :μ) => q_outμ,)`.
- `ctx`, the [`RuleContext`](@extref MessagePassingRulesBase.RuleContext) of
  [services](@extref MessagePassingRulesBase glossary-service) a rule reads, such as a random
  number generator. The table does not check its services.
- `logscale`, the [log scales](@extref MessagePassingRulesBase glossary-log-scale) that arrived
  with the messages, for a rule declared with `reads_logscale = true`, such as a mixture's:
  `logscale = (in = 0.5,)`.

The expected value is the rule's result.
[`approximately_equal`](@ref MessagePassingRulesTestUtils.approximately_equal) compares it with
what the rule returns. The type must match exactly, a `Normal{Float64}` for a `Normal{Float64}`,
and the values must agree within the tolerances.

Two wrappers check more than the result. An [`ExpectedWithLogScale`](@ref) checks the log scale a
message rule declares as well. An [`ExpectedWithAnnotations`](@ref) checks the annotations a rule
writes.

```@example tables
@testset "Gaussian: out, with its log scale" begin
    @test_message_update_rule(
        node = Gaussian, target = :out,
        cases = [
            (m = (μ = NormalMeanVariance(0.0, 1.0), v = PointMass(2.0)),) => ExpectedWithLogScale(NormalMeanVariance(0.0, 3.0), 0),
        ],
    )
end
nothing # hide
```

## Marginal rules and average energies

A marginal rule's table names a cluster as its target. This one expects the joint of `out` and
`μ`:

```@example tables
@testset "Gaussian: (out, μ)" begin
    @test_marginal_update_rule(
        node = Gaussian, target = (:out, :μ),
        cases = [
            (m = (out = NormalMeanVariance(1.0, 1.0), μ = NormalMeanVariance(0.0, 2.0)), q = (v = PointMass(0.5),)) =>
                MvNormalWeightedMeanPrecision([1.0, 0.0], [3.0 -2.0; -2.0 2.5]),
        ],
    )
end
nothing # hide
```

An average energy's table has no target, and it expects a number:

```@example tables
@testset "Gaussian: average energy" begin
    @test_average_energy(
        node = Gaussian,
        cases = [
            (q = (out = NormalMeanVariance(0.0, 1.0), μ = NormalMeanVariance(1.0, 1.0), v = PointMass(1.0)),) => 1.5 + log(2π) / 2,
        ],
    )
end
nothing # hide
```

## What every case checks

Beyond the result itself, each case checks:

- **Float-type promotion.** The case runs again with its inputs converted to `Float32`,
  `Float64` and `BigFloat`, the `float_types`. It converts all the inputs at once, then each
  alone. With `check_type_promotion = :exhaustive`, it converts every subset. The result must
  have the promoted type. A rule that hard-codes `Float64` fails, and so does a computed log
  scale that ignores the inputs' precision.
- **In-place rules.** A rule declared `inplace = true` runs again through `rule!`, into a buffer
  from its `preallocate`. It must write into that buffer, and the result must agree.
- **Scratch.** A rule declared with `scratch` runs again on a scratch that
  [`poison!`](@ref MessagePassingRulesTestUtils.poison!) filled with `NaN`. It must give the same
  result, so a rule that reads its scratch before writing it fails.
- **Allocations.** With `check_nonallocating = true`, the rule's call through the engine's entry
  point must allocate nothing.

The same average-energy case makes more checks when every subset of its three inputs is converted
and its allocations are measured. Compare the counts with the table above:

```@example tables
@testset "Gaussian: average energy, exhaustively" begin
    @test_average_energy(
        node = Gaussian,
        cases = [
            (q = (out = NormalMeanVariance(0.0, 1.0), μ = NormalMeanVariance(1.0, 1.0), v = PointMass(1.0)),) => 1.5 + log(2π) / 2,
        ],
        check_type_promotion = :exhaustive, check_nonallocating = true,
    )
end
nothing # hide
```

The promotion check shapes how a rule is written. The average energy computes `log(2v * π)`,
not `log(2π * v)`: `2π` is a `Float64`, which would turn a `Float32` energy into a `Float64` one,
and the promoted runs would fail.

The tolerances are `atol` and `rtol`. Each is a number, or a dictionary from float type to
number. By default, `atol` is `1e-4`, `1e-6` and `1e-8` for `Float32`, `Float64` and `BigFloat`,
and `rtol` is `0`. Every case also records the rule it selected, for
[the rule-coverage gate](@ref "The rule-coverage gate").

## API

The macros are the usual form. They report failures at the line of the table.

```@docs
@test_message_update_rule
@test_marginal_update_rule
@test_average_energy
```

The functions behind them take the node and the target positionally. Their docstrings describe
every keyword.

```@docs
test_message_update_rule
test_marginal_update_rule
test_average_energy
```

An expected value may carry more than the result.

```@docs
ExpectedWithLogScale
ExpectedWithAnnotations
```

The comparison the tables make, and the scratch poisoning, are public, for tests of your own.

```@docs
MessagePassingRulesTestUtils.approximately_equal
MessagePassingRulesTestUtils.poison!
```
