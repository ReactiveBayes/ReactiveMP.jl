@testitem "coverage:selected-rule" tags = [:testutils] setup = [Recording] begin
    using MessagePassingRulesTestUtils, MessagePassingRulesBase, BayesBase
    using MessagePassingRulesTestUtils: rule_test_locations

    module Covered
    using MessagePassingRulesBase, BayesBase
    struct Node end
    @define_factor_node(node = Node, type = Stochastic, interfaces = [:out, :x])
    # A broad rule, and a specific one that takes precedence for Float64 inputs.
    @define_message_update_rule(node = Node, target = :out, args = (m[:x]::PointMass,), body = (args) -> args.m[:x])
    @define_message_update_rule(node = Node, target = :out, args = (m[:x]::PointMass{Float64},), body = (args) -> args.m[:x])
    @define_message_update_rule(node = Node, target = :x, args = (m[:out]::PointMass,), body = (args) -> args.m[:out])

    struct Untouched end
    @define_factor_node(node = Untouched, type = Stochastic, interfaces = [:out])
    end

    @test length(check_rule_coverage(Covered)) == 5      # three rules, two nodes

    Recording.recorded() do
        @test_message_update_rule(node = Covered.Node, target = :out, check_type_promotion = false, cases = [(m = (x = PointMass(1.0),),) => PointMass(1.0)])
    end
    gaps = check_rule_coverage(Covered)
    messages = map(g -> g.message, gaps)
    # The Float64 case selected the specific rule; the broad one, which would also have
    # matched, stays uncovered.
    @test length(gaps) == 3
    @test count(m -> contains(m, "towards :out"), messages) == 1
    @test any(m -> contains(m, "towards :x"), messages)
    @test any(m -> contains(m, "Untouched"), messages)
    specific = only(filter(s -> s.target === MessagePassingRulesBase.Target{:out} && s.inputs[1].type === PointMass{Float64}, MessagePassingRulesBase.registered_rules(Covered)))
    @test endswith(String(only(rule_test_locations(specific)).file), "coverage_tests.jl")
end
