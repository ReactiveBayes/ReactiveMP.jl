include("discrete_transition/categoricals.jl")
include("discrete_transition/a.jl")
include("discrete_transition/marginals.jl")

function add_discrete_transition_rule_benchmarks(SUITE)
    SUITE["DiscreteTransition"] = BenchmarkGroup()
    add_discrete_transition_categoricals_rule_benchmarks(SUITE["DiscreteTransition"])
    add_discrete_transition_a_rule_benchmarks(SUITE["DiscreteTransition"])
    # add_discrete_transition_marginals_benchmarks(SUITE)
end

function add_rules_benchmarks(SUITE)
    SUITE["Rules"] = BenchmarkGroup()
    add_discrete_transition_rule_benchmarks(SUITE["Rules"])
end
