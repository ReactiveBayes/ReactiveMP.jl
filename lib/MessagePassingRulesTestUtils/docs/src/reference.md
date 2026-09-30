# Comparing with a reference

A rule may reimplement another implementation, such as a rule from an earlier release of a
package. Such a rule is compared with its reference on the same inputs.
[`compare_with_reference`](@ref) makes the comparison a test. It passes when the two agree, in
their results and their [log scales](@extref MessagePassingRulesBase glossary-log-scale). It fails
otherwise, unless the difference was investigated and declared.

## Compare a rule

The rule under test is the belief propagation rule towards `out` of the `Gaussian` node of the
[overview](@ref "A node to test"). The reference is a plain function with the same formula:

```@example reference
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

reference_out(μ, v) = (NormalMeanVariance(mean(μ), var(μ) + mean(v)), 0.0)

inputs = (m = (μ = NormalMeanVariance(1.0, 2.0), v = PointMass(0.5)),)
actual = @call_message_update_rule(node = Gaussian, target = :out, m = inputs.m)
reference, reference_logscale = reference_out(inputs.m.μ, inputs.m.v)

record = compare_with_reference(
    "Gaussian:out:m", getresult(actual), reference;
    node = Gaussian, target = :out, inputs,
    actual_logscale = getlogscale(actual), reference_logscale,
)
record.outcome
```

Each comparison returns a [`MigrationRecord`](@ref), which holds both sides and the outcome.
Here the outcome is `:agree`.

## Declare a disagreement

A [`DeclaredDisagreement`](@ref) records an investigated difference with its kind and its
reasoning:

- `:correction`: the reference is wrong, and the new implementation deliberately differs;
- `:migration_bug`: the new implementation is wrong, and awaits a fix.

Neither side is ever silently preferred. Every difference is either an agreement within the
tolerance or a declaration with its reasoning, which is logged whenever the difference is met.

Suppose the reference for the variational rule adds the variance of ``q(x)``, as the belief
propagation rule adds the variance of its message. The rule under test does not, and it is
right: a [variational](@extref MessagePassingRulesBase glossary-vmp) message reads only the mean
of ``q(x)``.

```@example reference
reference_vmp(μ, v) = NormalMeanVariance(mean(μ), var(μ) + mean(v))

q = (μ = NormalMeanVariance(1.0, 2.0), v = PointMass(0.5))
declared = [
    DeclaredDisagreement("Gaussian:out:q"; kind = :correction, reasoning = "the reference adds var(q(x)), which a variational message does not read"),
]

corrected = compare_with_reference(
    "Gaussian:out:q", getresult(@call_message_update_rule(node = Gaussian, target = :out, q = q)), reference_vmp(q.μ, q.v);
    node = Gaussian, target = :out, inputs = (; q), declared,
)
corrected.outcome
```

The comparison logs the declaration and returns its kind. Without the declaration, the same
comparison is a failing `Test` assertion that reports both sides.

## Save and load the records

[`save_migration_fixtures`](@ref) writes records to a file, and [`load_migration_fixtures`](@ref)
reads them back:

```@example reference
path = save_migration_fixtures(tempname(), [record, corrected]; packages = Dict("MyRules" => v"1.0.0"))
fixtures = load_migration_fixtures(path)
fixtures.header
```

The file keeps every value exactly, but only the Julia minor version that wrote it can read it.
For fixtures that other Julia versions read, see [Engine trajectories](@ref).

## API

```@docs
compare_with_reference
DeclaredDisagreement
MigrationRecord
save_migration_fixtures
load_migration_fixtures
```
