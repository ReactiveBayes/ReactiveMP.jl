"""
    MessagePassingRulesBase

The rule system of message passing on factor graphs: the macros that declare factor nodes,
their message update rules, marginal rules, average energies and dependencies, and the lookup
an engine uses to find and run a rule for a node, a target and the types of its inputs. Rules
are ordinary functions of their inputs, callable and testable without an engine, and found
through Julia's dispatch, whichever loaded package defines them.

- [`@define_factor_node`](@ref) declares a node: its interfaces, its kind and its algorithm;
- [`@define_message_update_rule`](@ref), [`@define_marginal_update_rule`](@ref) and
  [`@define_average_energy`](@ref) define what it computes;
- [`@define_dependencies`](@ref) declares what the rules consume under an algorithm of the
  node's own;
- [`call_message_update_rule`](@ref) and its siblings run a rule by hand, and
  [`which_message_update_rule`](@ref) and its siblings say which one would run.

It depends on BayesBase only, not on a distribution package or an engine.

# Examples

```jldoctest
julia> struct Shift end

julia> @define_factor_node(node = Shift, type = Deterministic, interfaces = [:out, :in])

julia> @define_message_update_rule(
           node = Shift, target = :out, args = (m[:in]::Real,), logscale = 0,
           body = (args) -> args.m[:in] + 1,
       )

julia> getresult(@call_message_update_rule(node = Shift, target = :out, m = (in = 1.0,)))
2.0
```
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
# Nodes: the declaration as data, and its queries.
@compat public NodeSpec, InterfaceSpec, nodespec, interfaces, interface_groups, sdtype, default_algorithm
@compat public static_inputs, matched_groups, min_group_length, required_factorisation, initial_messages
@compat public alias_interface, nodefunction
# Targets, and the inputs a rule receives.
@compat public Target, IndexedTarget, ClusterTarget, target_edge, target_index, cluster_members
@compat public RuleArgs, Messages, Marginals, RuleLogScales, canonical_cluster_keys, rule_inputs
@compat public RuleContext, DEFAULT_CONTEXT_SERVICES, buffer_like, cluster_blocks, check_factorized_cluster
@compat public RuleAnnotations, AnnotationStore, NoAnnotations, annotate!, hasannotation, getannotation
# Algorithms and dependency declarations, as an engine reads them.
@compat public ispure, DependenciesSpec, TargetDependencies, Dependency, dependencies_spec
@compat public target_dependencies, extends_default_scheme, free_energy_partition
@compat public DependencySelector, SingleInterface, AllGroupMembers, AlignedGroupMember, AllGroupMembersButSelf
@compat public CustomGroupSelector, select_group_members, selected_indices, selection_arity
# Resolving and running rules: what an engine calls.
@compat public RuleSpec, InputSpec, RuleNotFound, RuleNotFoundError
@compat public find_message_rule, find_marginal_rule, find_average_energy, rule_algorithm
@compat public execute_rule, execute_rule_with_logscale, rule_scratch
@compat public check_services, missing_services, check_reads_logscale
# Inspecting and checking the rules that exist.
@compat public Registry, registries, registered_rules, registered_nodes, registered_dependencies, duplicate_rules
@compat public list_rules, RuleCoverage, rule_coverage, check_rules, RuleIssue, check_rule_ambiguities
@compat public NodeFunctionLogPdf, visualize_spec

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
