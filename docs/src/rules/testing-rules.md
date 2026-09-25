# [Testing rules](@id rules-testing)

```@docs
MessagePassingRulesTestUtils
```

`MessagePassingRulesTestUtils` tests rules the way they are defined: on their own, without a
graph. Its tools are tables of cases, verification against a node's definition, comparison with
a reference implementation, and engine trajectories for whole graphs.

## [Tables](@id rules-testing-tables)

A table lists cases, each the inputs of one call and the expected result. The inputs are the
messages `m`, the marginals `q`, joint marginals `clusters` and a context `ctx`; a group is a
tuple in member order, with `nothing` where the rule does not take a member.

```julia
@test_message_update_rule(
    node = NormalMeanVariance, target = :out,
    cases = [
        (m = (μ = NormalMeanVariance(1.0, 2.0),), q = (v = PointMass(3.0),)) => NormalMeanVariance(1.0, 5.0),
        (q = (μ = PointMass(1.0), v = PointMass(2.0)),) => NormalMeanVariance(1.0, 2.0),
    ],
)
```

Each case is also run with its inputs converted to other float types (`float_types`), checking
that the result's type follows, unless `check_type_promotion = false`. Tolerances are `atol`
and `rtol`. A case that expects a log scale as well as a value gives an
[`ExpectedWithLogScale`](@ref); the log scale is checked on the case's inputs, and its float type
on every promoted run. A case for a rule that reads its inputs' log scales, such as a mixture's,
gives them as `logscale = (out = …,)` beside `m`. A case that expects annotations gives an
[`ExpectedWithAnnotations`](@ref). A rule with [scratch](@ref rules-defining-scratch) is also run on a
reused scratch that [`MessagePassingRulesTestUtils.poison!`](@ref) filled with NaN, and must give
the same result.

```@docs
@test_message_update_rule
@test_marginal_update_rule
@test_average_energy
test_message_update_rule
test_marginal_update_rule
test_average_energy
ExpectedWithLogScale
ExpectedWithAnnotations
check_rule_coverage
MessagePassingRulesTestUtils.RuleCoverageGap
MessagePassingRulesTestUtils.rule_test_locations
MessagePassingRulesTestUtils.approximately_equal
MessagePassingRulesTestUtils.poison!
```

## [Verification](@id rules-testing-verification)

A message rule can be checked against its node's definition: the message a rule computes must
be, up to a constant, the integral of the factor against its inputs, which the verification
computes numerically at a set of points. Derivatives of a rule's parameters can be checked
against finite differences.

```@docs
@verify_message_update_rule
verify_message_update_rule
verify_message_update
@test_rule_derivatives
test_rule_derivatives
```

## [Comparing with a reference](@id rules-testing-reference)

When a rule reimplements another implementation, the two are compared on the same inputs.
Declared disagreements, such as a correction to the reference, are recorded with their reasons,
so that a comparison agrees exactly where it should and nowhere else.

```@docs
compare_with_reference
DeclaredDisagreement
MigrationRecord
save_migration_fixtures
load_migration_fixtures
```

## [Engine trajectories](@id rules-testing-engine)

A whole graph is tested by its trajectory: the free energy per iteration, the final
posteriors, and every rule call in order with its result and log scale. Trajectories are saved
as TOML, portable across Julia versions, and compared call by call.

```@docs
EngineTrajectory
RuleCallRecord
compare_engine_trajectory
save_engine_fixture
load_engine_fixture
encode_fixture_value
```
