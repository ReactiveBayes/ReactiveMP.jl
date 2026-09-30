# MessagePassingRulesTestUtils

A rule package defines [factor nodes](@extref MessagePassingRulesBase glossary-factor-node) and
their [rules](@extref MessagePassingRulesBase glossary-rule) with
[`MessagePassingRulesBase`](@extref MessagePassingRulesBase.MessagePassingRulesBase). This package
tests those rules on their own, the way they are defined, without building a graph. It is a test
dependency only: nothing needs it at run time.

```@docs
MessagePassingRulesTestUtils
```

## A node to test

The examples on these pages test a toy node, a normal distribution with a known variance,
``f(y, x, v) = \mathcal{N}(y \mid x, v)``, where `y` is the output `out`, `x` the mean `μ` and `v`
the variance. Its messages are distributions from
[ExponentialFamily](https://github.com/ReactiveBayes/ExponentialFamily.jl).
[Your first node](@extref MessagePassingRulesBase tutorial-first-node) derives each of its rules.

```@example index
using MessagePassingRulesBase, MessagePassingRulesTestUtils
using BayesBase, ExponentialFamily, Distributions, Test

struct Gaussian end
Gaussian(μ, v) = NormalMeanVariance(μ, v)

@define_factor_node(node = Gaussian, type = Stochastic, interfaces = [:out, :μ, :v])

# Belief propagation towards `out`: the variances add.
@define_message_update_rule(
    node = Gaussian, target = :out,
    args = (m[:μ]::NormalMeanVariance, m[:v]::PointMass),
    logscale = 0,
    body = (args) -> NormalMeanVariance(mean(args.m[:μ]), var(args.m[:μ]) + mean(args.m[:v])),
)

# Variational message passing towards `out`: only the mean of q(x) matters.
@define_message_update_rule(
    node = Gaussian, target = :out,
    args = (q[:μ]::Any, q[:v]::PointMass),
    body = (args) -> NormalMeanVariance(mean(args.q[:μ]), mean(args.q[:v])),
)

# Belief propagation towards `μ` from an observed `out`.
@define_message_update_rule(
    node = Gaussian, target = :μ,
    args = (m[:out]::PointMass, m[:v]::PointMass),
    logscale = 0,
    body = (args) -> NormalMeanVariance(mean(args.m[:out]), mean(args.m[:v])),
)
nothing # hide
```

The line `Gaussian(μ, v) = NormalMeanVariance(μ, v)` makes the node callable as the distribution
of its output. The declaration then defines the node's log-density,
[`nodefunction`](@extref MessagePassingRulesBase.nodefunction), which
[verification](@ref "Verification against the node") integrates.

## Testing a rule package, end to end

A rule package's suite is a set of `@testitem`s, run by TestItemRunner, one or a few per node,
and a gate after them. Every rule package of this repository, such as
`StandardMessagePassingRules`, follows this recipe. Every check is a `Test` assertion, so the
tools go inside a `@testitem` or a `@testset`; the examples below use `@testset`.

### 1. A table for every rule

A table states, case by case, what a rule must return. A case is the inputs of one call and the
expected result. The inputs are `m` for the [messages](@extref MessagePassingRulesBase glossary-message),
`q` for the [marginals](@extref MessagePassingRulesBase glossary-marginal) and `clusters` for
joint marginals.

```@example index
@testset "Gaussian: out" begin
    @test_message_update_rule(
        node = Gaussian, target = :out,
        cases = [
            (m = (μ = NormalMeanVariance(1.0, 2.0), v = PointMass(0.5)),) => ExpectedWithLogScale(NormalMeanVariance(1.0, 2.5), 0),
            (q = (μ = NormalMeanVariance(1.0, 2.0), v = PointMass(0.5)),) => NormalMeanVariance(1.0, 0.5),
            (q = (μ = PointMass(-1.0), v = PointMass(2.0)),) => NormalMeanVariance(-1.0, 2.0),
        ],
    )
end
nothing # hide
```

The summary counts many more checks than cases, because the table runs every case again with
its inputs converted to other float types. It also checks an in-place rule through `rule!` and a rule with scratch on a
poisoned scratch. [Table tests](@ref) describes every check. Marginal rules and average
energies have tables of their own, [`@test_marginal_update_rule`](@ref) and
[`@test_average_energy`](@ref).

### 2. Verification against the node

A table's expected values are worked out by hand, and can share the author's mistake. For a
[stochastic node](@extref MessagePassingRulesBase glossary-stochastic-node),
[`@verify_message_update_rule`](@ref) checks a message rule against the node's own log-density,
integrated numerically. It checks the rule's [log scale](@extref MessagePassingRulesBase glossary-log-scale)
as well.

```@example index
@testset "Gaussian: verification" begin
    @verify_message_update_rule(node = Gaussian, target = :out, m = (μ = NormalMeanVariance(0.5, 1.5), v = PointMass(2.0)))
    @verify_message_update_rule(node = Gaussian, target = :μ, m = (out = PointMass(3.0), v = PointMass(0.5)))
end
nothing # hide
```

[Verification against the node](@ref) lists what it can integrate.

### 3. Derivatives, where they matter

A model may be differentiated through its rules with ForwardDiff.
[`@test_rule_derivatives`](@ref) checks such a rule against finite differences:

```@example index
@testset "Gaussian: derivatives" begin
    @test_rule_derivatives(
        node = Gaussian, target = :out,
        inputs = θ -> (m = (μ = NormalMeanVariance(θ, 1.0), v = PointMass(θ^2)),),
        at = 1.5, summary = d -> mean(d) + var(d),
    )
end
nothing # hide
```

[Derivatives](@ref) says what fails.

### 4. The rule-coverage gate

After the test items, [`check_rule_coverage`](@ref) lists every rule of the package that no test
selected. The tests above selected all three rules of the node, so the list is empty:

```@example index
check_rule_coverage(@__MODULE__)
```

A package's `test/runtests.jl` runs the gate only when its filter left no test item out, and a
new rule without a test then fails the suite. [The rule-coverage gate](@ref) shows that file and
says what counts as tested.

### Beyond rules

Two more tools serve a package that reimplements another:

- [`compare_with_reference`](@ref) compares a rule with a reference implementation on the same
  inputs, and each investigated difference is declared and explained. See
  [Comparing with a reference](@ref).
- An [`EngineTrajectory`](@ref) records a whole inference run: its free energy, its posteriors
  and every rule call. It is compared with a recorded run. See [Engine trajectories](@ref).

The rules on these pages are called directly, never inside a model. To use a node in a model, see
[ReactiveMP](https://reactivebayes.github.io/ReactiveMP.jl/dev/),
[GraphPPL](https://reactivebayes.github.io/GraphPPL.jl/stable/) and
[RxInfer](https://reactivebayes.github.io/RxInfer.jl/stable/).

## Pages

- [Table tests](@ref): tables of cases for message rules, marginal rules and average energies.
- [Verification against the node](@ref): a message rule against the node's log-density.
- [Derivatives](@ref): automatic differentiation through a rule.
- [The rule-coverage gate](@ref): every rule selected by some test.
- [Comparing with a reference](@ref): a rule against another implementation.
- [Engine trajectories](@ref): whole inference runs, recorded and compared.
