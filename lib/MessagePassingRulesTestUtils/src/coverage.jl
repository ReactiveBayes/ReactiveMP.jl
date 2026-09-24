const SELECTED_RULES = IdDict{Any, Vector{LineNumberNode}}()

record_selected_rule!(spec, source::LineNumberNode) = (push!(get!(SELECTED_RULES, spec, LineNumberNode[]), source); nothing)

# A rule a test calls by hand, with `call_message_update_rule` and its siblings, is recorded as
# selected too; it has no source line to report.
const DIRECT_CALL = LineNumberNode(0, Symbol("a direct call"))
record_direct_call!(spec) = record_selected_rule!(spec, DIRECT_CALL)

"""
    RuleCoverageGap

Something [`check_rule_coverage`](@ref) found untested: a rule no test case or direct call selected, or
a node none of whose rules was selected.
"""
struct RuleCoverageGap
    subject::Any
    message::String
end

Base.show(io::IO, gap::RuleCoverageGap) = print(io, "RuleCoverageGap(", gap.message, ")")

"""
    check_rule_coverage(modules...)

The rules and nodes defined in `modules` that no test exercised: no table case,
verification or derivative check, and no direct call such as
[`call_message_update_rule`](@ref), selected them. A rule counts as tested only when a test
actually *selected* it, so a broader rule answering in its place
leaves it reported.

Call it after the whole suite has run, and only when nothing was filtered out:

```julia
@run_package_tests
isempty(ARGS) && @test isempty(check_rule_coverage(MyRules))
```
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
    rule_test_locations(spec)

Where the table cases that selected `spec` were written.
"""
rule_test_locations(spec) = get(SELECTED_RULES, spec, LineNumberNode[])
