# The rule-coverage gate

A rule package ends its suite with a gate: after every test item has run, each rule the package
defines must have been selected by some test, and each node must have some rule selected. A rule
added without a test fails the suite.

A rule counts as tested only when a test *selected* it, resolved it as the rule to run, through:

- a table case, [`@test_message_update_rule`](@ref) and its siblings;
- a verification, [`@verify_message_update_rule`](@ref);
- a derivative check, [`@test_rule_derivatives`](@ref);
- a direct call, [`call_message_update_rule`](@extref MessagePassingRulesBase.call_message_update_rule),
  its siblings or their macros.

A broader rule that would also have answered a case stays reported, so every method of a rule
needs a case of its own. A rule reached only through a graph is not counted, since an engine
resolves rules itself.

Selections accumulate over the whole process, so the gate means something only after an
unfiltered run: `test/runtests.jl` notes whether its filter left any item out, and checks the
gate only when none was. A suite that tags items `:slow` and skips them by default runs the gate
only with them, `TEST_ALL=true`.

```julia
const FILTERED_OUT = Ref(false)
test_item_filter(ti) = is_selected(ti) || (FILTERED_OUT[] = true; false)

@run_package_tests(filter = test_item_filter, verbose = true)

if !FILTERED_OUT[]
    @testset "rule coverage" begin
        gaps = check_rule_coverage(MyRules, Base.get_extension(MyRules, :MyRulesExt))
        foreach(println, gaps)
        @test isempty(gaps)
    end
end
```

A package extension registers its rules in its own module, which the gate is given as well.

## API

```@docs
check_rule_coverage
MessagePassingRulesTestUtils.RuleCoverageGap
MessagePassingRulesTestUtils.rule_test_locations
```
