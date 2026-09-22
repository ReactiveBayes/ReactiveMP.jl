module MessagePassingRulesBase

using BayesBase, LinearAlgebra, MacroTools, TupleTools

export message_passing_rule, message_passing_rule!
export message_passing_marginalrule, message_passing_marginalrule!
export message_passing_average_energy
export @define_factor_node, Stochastic, Deterministic
export @define_message_update_rule, @define_marginal_update_rule, @define_average_energy
export @define_dependencies

include("targets.jl")
include("containers.jl")
include("annotations.jl")
include("algorithms.jl")
include("context.jl")
include("rulespec.jl")
include("macrohelpers.jl")
include("nodes.jl")
include("dependencies.jl")
include("registry.jl")
include("node_macro.jl")
include("rule_macro.jl")

end
