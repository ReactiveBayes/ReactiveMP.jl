# Table tests

A table lists cases, each the inputs of one call to a rule and the result the rule must return.
It is the main test of every rule: cheap to write, one line per case, and each case is checked in
several ways at once.

```julia
@test_message_update_rule(
    node = NormalMeanVariance, target = :out,
    cases = [
        (m = (μ = NormalMeanVariance(1.0, 2.0),), q = (v = PointMass(3.0),)) => NormalMeanVariance(1.0, 5.0),
        (q = (μ = PointMass(1.0), v = PointMass(2.0)),) => NormalMeanVariance(1.0, 2.0),
    ],
)
```

## Cases

A case is `inputs => expected`. The inputs are a `NamedTuple` of:

- `m`, the inbound messages, and `q`, the marginals of single interfaces, each keyed by the
  interfaces' declared names. A group is a tuple of its members in order, with `nothing` where
  the rule does not take a member: `m = (T = (PointMass(2.0), nothing),)`.
- `clusters`, the joint marginals of clusters, as pairs from the members to the joint:
  `clusters = ((:out, :μ) => q_outμ,)`.
- `ctx`, the [`RuleContext`](@extref MessagePassingRulesBase.RuleContext) of services a rule
  reads, such as a random number generator. Its services are not checked.
- `logscale`, the log scales that arrived with the messages, for a rule declared with
  `reads_logscale = true`, such as a mixture's: `logscale = (in = 0.5,)`.

The expected value is the rule's result, compared by
[`approximately_equal`](@ref MessagePassingRulesTestUtils.approximately_equal): the type must
match exactly, a `Normal{Float64}` for a `Normal{Float64}`, and the values within the tolerances.
A case that also checks the log scale a message rule declares gives an
[`ExpectedWithLogScale`](@ref); one that checks the annotations a rule writes gives an
[`ExpectedWithAnnotations`](@ref).

## What every case checks

Beyond the result itself:

- **Float-type promotion.** Each case runs again with its inputs converted to `Float32`,
  `Float64` and `BigFloat` (`float_types`): all of them at once, then each alone, or every subset
  with `check_type_promotion = :exhaustive`. The result must have the promoted type, so a rule
  that hard-codes `Float64` fails, and so does a computed log scale that ignores the inputs'
  precision.
- **In-place rules.** A rule declared `inplace = true` runs again through `rule!` into a buffer
  from its `preallocate`, and must write into that buffer and agree.
- **Scratch.** A rule declared with `scratch` runs again on a scratch after
  [`poison!`](@ref MessagePassingRulesTestUtils.poison!) filled it with `NaN`, and must give the
  same result: a rule that reads its scratch before writing it fails.
- **Allocations.** With `check_nonallocating = true`, the rule's call through the engine's entry
  point must allocate nothing.

Tolerances are `atol` and `rtol`, each a number or a dictionary from float type to number; by
default `atol` is `1e-4`, `1e-6` and `1e-8` for `Float32`, `Float64` and `BigFloat`, and `rtol` is
`0`. Every case also records the rule it selected, for [The rule-coverage gate](@ref).

## API

The macros are the usual form: they report failures at the line of the table.

```@docs
@test_message_update_rule
@test_marginal_update_rule
@test_average_energy
```

The functions behind them take the node and the target positionally, and describe every
keyword.

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

The comparison the tables make, and the scratch poisoning, are public for tests of their own.

```@docs
MessagePassingRulesTestUtils.approximately_equal
MessagePassingRulesTestUtils.poison!
```
