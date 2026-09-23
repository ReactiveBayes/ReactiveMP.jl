module MessagePassingRulesTestUtils

using BayesBase, Distributions, ForwardDiff, HCubature, Serialization, Statistics, Test
using MessagePassingRulesBase

export @test_message_update_rule, @test_marginal_update_rule, @test_average_energy
export test_message_update_rule, test_marginal_update_rule, test_average_energy
export ExpectedWithAnnotations
export check_rule_coverage
export @verify_message_update_rule, verify_message_update_rule, verify_message_update
export @test_rule_derivatives, test_rule_derivatives

include("checks.jl")
include("coverage.jl")
include("table_tests.jl")
include("verification.jl")
include("derivatives.jl")

end
