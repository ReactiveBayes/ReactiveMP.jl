module MessagePassingRulesTestUtils

using BayesBase, Distributions, ForwardDiff, HCubature, Serialization, Statistics, Test
using MessagePassingRulesBase

export @test_message_update_rule, @test_marginal_update_rule, @test_average_energy
export test_message_update_rule, test_marginal_update_rule, test_average_energy
export ExpectedWithAnnotations

include("checks.jl")
include("coverage.jl")
include("table_tests.jl")

end
