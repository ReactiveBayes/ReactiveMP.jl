using BenchmarkTools

const SUITE = BenchmarkGroup()

include("rules/rules.jl")

add_rules_benchmarks(SUITE)
