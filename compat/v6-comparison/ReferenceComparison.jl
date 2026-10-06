# Compares a v7 rule's result with v6's on the same inputs (`compare_with_reference`), and whole
# inference runs with the ones recorded from v6 (`EngineTrajectory`, `fixtures/`). It reports
# through MessagePassingRulesTestUtils' checks, and goes with this environment at the release.
module ReferenceComparison

using BayesBase, Distributions, Serialization, Test, TOML
using MessagePassingRulesBase: FactorizedCluster, cluster_blocks
using MessagePassingRulesTestUtils: approximately_equal, record_check

export compare_with_reference, DeclaredDisagreement, MigrationRecord, save_migration_fixtures, load_migration_fixtures
export encode_fixture_value, RuleCallRecord, EngineTrajectory, save_engine_fixture, load_engine_fixture, compare_engine_trajectory

include(joinpath("reference", "migration.jl"))
include(joinpath("reference", "engine_fixtures.jl"))

end
