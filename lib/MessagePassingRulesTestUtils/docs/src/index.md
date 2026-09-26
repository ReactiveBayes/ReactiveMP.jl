# MessagePassingRulesTestUtils

A rule package, one that defines nodes and rules with
[`MessagePassingRulesBase`](@extref MessagePassingRulesBase.MessagePassingRulesBase), tests its rules
with this package: on their own, the way they are defined, without building a graph. It is a test
dependency only; nothing at run time needs it.

```@docs
MessagePassingRulesTestUtils
```

## Testing a rule package, end to end

A rule package's suite is a set of `@testitem`s run by TestItemRunner, one or a few per node, and
a gate after them. The rule packages of this repository, such as `StandardMessagePassingRules`,
all follow this recipe.

### 1. A table for every rule

Each test item states, case by case, what a rule must return. A case is the inputs of one call,
`m` for messages, `q` for marginals, `clusters` for joint marginals, and the expected result:

```julia
@testitem "rules:NormalMeanVariance:out" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions

    @test_message_update_rule(
        node = NormalMeanVariance, target = :out,
        cases = [
            (m = (μ = PointMass(-1.0), v = PointMass(2.0)),) => NormalMeanVariance(-1.0, 2.0),
            (m = (μ = NormalMeanVariance(0.0, 1.0), v = PointMass(2.0)),) => ExpectedWithLogScale(NormalMeanVariance(0.0, 3.0), 0),
            (q = (μ = NormalMeanVariance(1.0, 2.0), v = InverseGamma(3.0, 4.0)),) => NormalMeanVariance(1.0, 4 / 3),
        ],
    )
end
```

The table runs every case again with its inputs converted to other float types, checks an
in-place rule through `rule!`, and a rule with scratch on a poisoned scratch; see
[Table tests](@ref). Marginal rules and average energies have their own tables,
[`@test_marginal_update_rule`](@ref) and [`@test_average_energy`](@ref).

### 2. Verification against the node

A table's expected values are worked out by hand, and can share the author's mistake. For a
stochastic node, [`@verify_message_update_rule`](@ref) checks a message rule against the node's
own log-density instead, integrated numerically, and the rule's log scale with it:

```julia
@verify_message_update_rule(node = NormalMeanVariance, target = :out, m = (μ = NormalMeanVariance(0.5, 1.5), v = PointMass(2.0)))
@verify_message_update_rule(node = NormalMeanVariance, target = :μ, q = (out = NormalMeanVariance(1.0, 2.0), v = InverseGamma(3.0, 4.0)))
```

See [Verification against the node](@ref) for what it can integrate.

### 3. Derivatives, where they matter

A rule that a model differentiates through, with ForwardDiff, is checked with
[`@test_rule_derivatives`](@ref) against finite differences; see [Derivatives](@ref).

### 4. The rule-coverage gate

`test/runtests.jl` runs the items and then, only when none was filtered out, asks
[`check_rule_coverage`](@ref) for every rule of the package that no test selected. A new rule
without a test fails the suite:

```julia
using TestItemRunner, Test, MessagePassingRulesTestUtils, MyRules

# `is_selected(ti)` is the suite's own selection, from `ARGS` (paths, `tag:`, `name:`).
const FILTERED_OUT = Ref(false)
test_item_filter(ti) = is_selected(ti) || (FILTERED_OUT[] = true; false)

@run_package_tests(filter = test_item_filter, verbose = true)

if !FILTERED_OUT[]
    @testset "rule coverage" begin
        gaps = check_rule_coverage(MyRules)
        foreach(println, gaps)
        @test isempty(gaps)
    end
end
```

See [The rule-coverage gate](@ref) for what counts as tested.

### Beyond rules

Two more tools serve a package that reimplements another:
[`compare_with_reference`](@ref) compares a rule with a reference implementation on the same
inputs, with investigated differences declared and explained ([Comparing with a
reference](@ref)); an [`EngineTrajectory`](@ref) records a whole inference run, free energy,
posteriors and every rule call, and compares it with a recorded one ([Engine
trajectories](@ref)).

## Pages

- [Table tests](@ref): tables of cases for message rules, marginal rules and average energies.
- [Verification against the node](@ref): a message rule against the node's log-density.
- [Derivatives](@ref): automatic differentiation through a rule.
- [The rule-coverage gate](@ref): every rule selected by some test.
- [Comparing with a reference](@ref): a rule against another implementation.
- [Engine trajectories](@ref): whole inference runs, recorded and compared.
