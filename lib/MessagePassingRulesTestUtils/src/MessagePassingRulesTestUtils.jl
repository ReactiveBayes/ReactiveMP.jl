"""
    MessagePassingRulesTestUtils

Tools for testing message passing rules on their own, without a graph, the way a rule package
defines them with [`@define_message_update_rule`](@extref MessagePassingRulesBase.@define_message_update_rule)
and its siblings. Every check is a `Test` assertion, reported against the line of the test that
made it, so the tools go inside ordinary `@testset`s or `@testitem`s.

A rule package's suite uses them in four layers:

1. **Table tests**, [`@test_message_update_rule`](@ref), [`@test_marginal_update_rule`](@ref)
   and [`@test_average_energy`](@ref): each case gives a rule's inputs and the result it must
   return. Every case is also run with its inputs converted to other float types, through
   `rule!` for an in-place rule and on a poisoned scratch for a rule with scratch.
2. **Verification against the node**, [`@verify_message_update_rule`](@ref): a message rule's
   output, and its log scale, against the node's own log-density, integrated numerically.
3. **Derivative checks**, [`@test_rule_derivatives`](@ref): automatic differentiation through
   a rule against finite differences.
4. **The rule-coverage gate**, [`check_rule_coverage`](@ref): after an unfiltered run of the
   suite, every rule the package defines must have been selected by some test.

For implementations with a predecessor, [`compare_with_reference`](@ref) compares a rule with a
reference implementation on the same inputs, and [`EngineTrajectory`](@ref) records and
compares whole inference runs.

# Examples

```jldoctest; setup = :(using MessagePassingRulesBase, BayesBase, Distributions)
julia> struct Gauss  # a toy node: out ~ Normal(μ, σ)
           μ::Float64
           σ::Float64
       end

julia> BayesBase.logpdf(d::Gauss, x) = logpdf(Normal(d.μ, d.σ), x);

julia> @define_factor_node(node = Gauss, type = Stochastic, interfaces = [:out, :μ, :σ]);

julia> @define_message_update_rule(
           node = Gauss, target = :out, args = (m[:μ]::PointMass, m[:σ]::PointMass),
           body = (args) -> Normal(mean(args.m[:μ]), mean(args.m[:σ])),
       );

julia> @test_message_update_rule(
           node = Gauss, target = :out,
           cases = [(m = (μ = PointMass(1.0), σ = PointMass(2.0)),) => Normal(1.0, 2.0)],
       )

julia> @verify_message_update_rule(node = Gauss, target = :out, m = (μ = PointMass(1.0), σ = PointMass(2.0)));

julia> isempty(check_rule_coverage(@__MODULE__))
true
```
"""
module MessagePassingRulesTestUtils

using BayesBase, Distributions, ForwardDiff, HCubature, Serialization, Statistics, Test, TOML
using Compat: @compat
using MessagePassingRulesBase
using MessagePassingRulesBase: FactorizedCluster, cluster_blocks

export @test_message_update_rule, @test_marginal_update_rule, @test_average_energy
export test_message_update_rule, test_marginal_update_rule, test_average_energy
export ExpectedWithAnnotations, ExpectedWithLogScale
export check_rule_coverage
export @verify_message_update_rule, verify_message_update_rule, verify_message_update
export @test_rule_derivatives, test_rule_derivatives
export compare_with_reference, DeclaredDisagreement, MigrationRecord, save_migration_fixtures, load_migration_fixtures
export encode_fixture_value, RuleCallRecord, EngineTrajectory, save_engine_fixture, load_engine_fixture, compare_engine_trajectory
# Named on the documentation and useful in a test, but generic enough to clash if exported.
@compat public poison!, approximately_equal, RuleCoverageGap, rule_test_locations

include("checks.jl")
include("coverage.jl")
include("table_tests.jl")
include("verification.jl")
include("derivatives.jl")
include("migration.jl")
include("engine_fixtures.jl")

function __init__()
    push!(MessagePassingRulesBase.INTERACTIVE_SELECTION_OBSERVERS, record_direct_call!)
    return nothing
end

end
