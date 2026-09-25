"""
    MessagePassingRulesTestUtils

Tools for testing rules without a graph: tables of cases (`@test_message_update_rule`),
verification against a node's definition, comparison with a reference implementation, the
rule-coverage gate, and engine trajectories for whole graphs.
"""
module MessagePassingRulesTestUtils

using BayesBase, Distributions, ForwardDiff, HCubature, Serialization, Statistics, Test, TOML
using MessagePassingRulesBase
using MessagePassingRulesBase: FactorizedCluster, cluster_blocks

export @test_message_update_rule, @test_marginal_update_rule, @test_average_energy
export test_message_update_rule, test_marginal_update_rule, test_average_energy
export ExpectedWithAnnotations
export check_rule_coverage
export @verify_message_update_rule, verify_message_update_rule, verify_message_update
export @test_rule_derivatives, test_rule_derivatives
export compare_with_reference, DeclaredDisagreement, MigrationRecord, save_migration_fixtures, load_migration_fixtures
export encode_fixture_value, RuleCallRecord, EngineTrajectory, save_engine_fixture, load_engine_fixture, compare_engine_trajectory

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
