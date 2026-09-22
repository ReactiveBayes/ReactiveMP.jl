module MessagePassingRulesBase

using BayesBase, LinearAlgebra, MacroTools, TupleTools

export message_passing_rule, message_passing_rule!
export message_passing_marginalrule, message_passing_marginalrule!
export message_passing_average_energy
export @define_factor_node, Stochastic, Deterministic
export @define_message_update_rule, @define_marginal_update_rule, @define_average_energy
export @define_dependencies
export @call_rule, @call_marginalrule, @call_average_energy, @which_rule
export call_rule, call_marginalrule, call_average_energy, which_rule

include("targets.jl")
include("containers.jl")
include("annotations.jl")
include("algorithms.jl")
include("context.jl")
include("buffers.jl")
include("rulespec.jl")
include("macrohelpers.jl")
include("nodes.jl")
include("dependencies.jl")
include("registry.jl")
include("node_macro.jl")
include("rule_macro.jl")
include("diagnostics.jl")
include("interactive.jl")
include("display.jl")

function __init__()
    Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, kwargs
        exc.f === visualize_spec &&
            print(io, "\n`visualize_spec` needs a visualisation backend, provided by a package extension, and none is loaded.")
    end
    return nothing
end

end
