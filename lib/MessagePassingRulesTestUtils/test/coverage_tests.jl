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

@testitem "coverage:direct-calls" tags = [:testutils] begin
    using MessagePassingRulesTestUtils, MessagePassingRulesBase, BayesBase

    module CalledByHand
    using MessagePassingRulesBase, BayesBase
    struct Node end
    @define_factor_node(node = Node, type = Stochastic, interfaces = [:out, :x])
    @define_message_update_rule(node = Node, target = :out, args = (m[:x]::PointMass,), body = (args) -> args.m[:x])
    @define_marginal_update_rule(node = Node, target = (:out, :x), args = (m[:out]::PointMass, m[:x]::PointMass), body = (args) -> args.m[:out])
    @define_average_energy(node = Node, args = (q[:out]::PointMass, q[:x]::PointMass), body = (args) -> 0.0)
    end
    N = CalledByHand

    @test length(check_rule_coverage(N)) == 4      # three rules, one node
    # A rule a test runs by hand counts as tested, as a table case does: the interactive calls
    # report the rule they selected.
    @test call_message_update_rule(N.Node, :out; m = (x = PointMass(1.0),)) == PointMass(1.0)
    @test call_marginal_update_rule(N.Node, (:out, :x); m = (out = PointMass(1.0), x = PointMass(2.0))) == PointMass(1.0)
    @test length(check_rule_coverage(N)) == 1
    @test call_average_energy(N.Node; q = (out = PointMass(1.0), x = PointMass(2.0))) == 0.0
    @test isempty(check_rule_coverage(N))
end
