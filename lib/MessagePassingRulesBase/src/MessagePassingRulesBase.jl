"""
    MessagePassingRulesBase

The rule system: the macros that declare factor nodes, their message update rules, marginal
rules, average energies and dependencies, and the lookup an engine uses to find and run a rule
for a node, a target and the types of its inputs. Rules are ordinary functions of their inputs,
callable and testable without an engine.
"""
module MessagePassingRulesBase

using BayesBase, MacroTools
using Compat: @compat

export message_passing_rule, message_passing_rule!
export message_passing_marginalrule, message_passing_marginalrule!
export message_passing_average_energy
export @define_factor_node, Stochastic, Deterministic, getnodefn
export AbstractAlgorithm, DefaultAlgorithm, DefaultAlgorithmExtension
export FactorizedCluster, public_equivalent, matrix_correction
export @define_message_update_rule, @define_marginal_update_rule, @define_average_energy
export @define_dependencies
export NodeFunctionRuleFallback
export @call_message_update_rule, @call_marginal_update_rule, @call_average_energy
export @which_message_update_rule, @which_marginal_update_rule, @which_average_energy
export call_message_update_rule, call_marginal_update_rule, call_average_energy
export which_message_update_rule, which_marginal_update_rule, which_average_energy
export UndefinedLogScale, UndefinedLogScaleError, require_logscale, isdefined_logscale
export with_logscale, from_body, getlogscale
export RuleResult, getresult, getrule, getannotations
# Generic names a downstream package may well define for itself: public, not exported.
@compat public getalgorithm, getcontext, getscratch, getarguments, gettarget

include("targets.jl")
include("factorized_cluster.jl")
include("public_equivalent.jl")
include("containers.jl")
include("logscale.jl")
include("annotations.jl")
include("algorithms.jl")
include("context.jl")
include("buffers.jl")
include("rulespec.jl")
include("result.jl")
include("macrohelpers.jl")
include("nodes.jl")
include("dependencies.jl")
include("registry.jl")
include("node_macro.jl")
include("rule_macro.jl")
include("diagnostics.jl")
include("fallback.jl")
include("interactive.jl")
include("display.jl")
include("result_show.jl")

function __init__()
    Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, kwargs
        exc.f === visualize_spec &&
            print(io, "\n`visualize_spec` needs a visualisation backend, provided by a package extension, and none is loaded.")
    end
    return nothing
end

end
