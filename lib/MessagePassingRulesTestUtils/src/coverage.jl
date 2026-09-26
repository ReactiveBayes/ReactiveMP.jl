const SELECTED_RULES = IdDict{Any, Vector{LineNumberNode}}()

record_selected_rule!(spec, source::LineNumberNode) = (push!(get!(SELECTED_RULES, spec, LineNumberNode[]), source); nothing)

# A rule a test calls by hand, with `call_message_update_rule` and its siblings, is recorded as
# selected too; it has no source line to report.
const DIRECT_CALL = LineNumberNode(0, Symbol("a direct call"))
record_direct_call!(spec) = record_selected_rule!(spec, DIRECT_CALL)

"""
    RuleCoverageGap

Something [`check_rule_coverage`](@ref) found untested: a rule that no test selected, or a node
none of whose rules any test selected. `subject` is the rule's
[`RuleSpec`](@extref MessagePassingRulesBase.RuleSpec) or the node's
[`NodeSpec`](@extref MessagePassingRulesBase.NodeSpec); `message` says what is missing, as in
`"no test selects …"`, and is what the gap prints.
"""
struct RuleCoverageGap
    subject::Any
    message::String
end

Base.show(io::IO, gap::RuleCoverageGap) = print(io, "RuleCoverageGap(", gap.message, ")")

"""
    check_rule_coverage(modules...) -> Vector{RuleCoverageGap}

List the rules and nodes defined in `modules` that no test exercised, as
[`RuleCoverageGap`](@ref)s; an empty list means every rule was tested.

A rule counts as tested only when some test *selected* it, that is resolved it as the rule to
run: a table case, a verification ([`verify_message_update_rule`](@ref)), a derivative check
([`test_rule_derivatives`](@ref)), or a direct call through
[`call_message_update_rule`](@extref MessagePassingRulesBase.call_message_update_rule), its
siblings or their macros. A broader rule that would also have answered stays reported, and so
does a rule reached only through a graph, since an engine resolves rules itself. A node counts
as tested when any of its rules in `modules` was selected.

Selections accumulate in the process from the moment the package is loaded and are never
reset, so the gate means something only after the whole suite has run in one process, with
nothing filtered out. A package's `test/runtests.jl` calls it after TestItemRunner's
`@run_package_tests`, and only when no test item was left out:

```julia
# `is_selected(ti)` is the suite's own selection, from `ARGS`.
const FILTERED_OUT = Ref(false)
test_item_filter(ti) = is_selected(ti) || (FILTERED_OUT[] = true; false)

@run_package_tests(filter = test_item_filter)

if !FILTERED_OUT[]
    @testset "rule coverage" begin
        gaps = check_rule_coverage(MyRules)
        foreach(println, gaps)
        @test isempty(gaps)
    end
end
```

# Arguments

- `modules`: the modules whose rules and nodes to check, such as the package and its
  extensions; with none, every loaded module.

See also [`rule_test_locations`](@ref).
"""
function check_rule_coverage(modules::Module...)
    gaps = RuleCoverageGap[]
    specs = MessagePassingRulesBase.registered_rules(modules...)
    for spec in specs
        haskey(SELECTED_RULES, spec) && continue
        push!(gaps, RuleCoverageGap(spec, "no test selects $(repr(spec))"))
    end
    for node in MessagePassingRulesBase.registered_nodes(modules...)
        any(spec -> spec.node === node.node && haskey(SELECTED_RULES, spec), specs) && continue
        push!(gaps, RuleCoverageGap(node, "no test exercises any rule of $(node.node)"))
    end
    return gaps
end

"""
    rule_test_locations(spec) -> Vector{LineNumberNode}

Where the tests that selected the rule `spec` were written: one `LineNumberNode` per selection,
from a table case, a verification or a derivative check, in the order they ran. A direct call
through [`call_message_update_rule`](@extref MessagePassingRulesBase.call_message_update_rule)
and its siblings has no source line and appears as `LineNumberNode(0, Symbol("a direct call"))`.
Empty when no test selected `spec`.
"""
rule_test_locations(spec) = get(SELECTED_RULES, spec, LineNumberNode[])
