@testitem "quality:aqua" tags = [:quality] begin
    using Aqua, DiscreteTransitionMessagePassingRules
    Aqua.test_all(DiscreteTransitionMessagePassingRules)
end

@testitem "quality:no ambiguities" tags = [:quality] begin
    using DiscreteTransitionMessagePassingRules, Test
    @test isempty(Test.detect_ambiguities(DiscreteTransitionMessagePassingRules; recursive = true))
end

@testitem "quality:rules" tags = [:quality] begin
    using DiscreteTransitionMessagePassingRules, Test
    using MessagePassingRulesBase: check_rules, check_rule_ambiguities
    @test isempty(check_rules(DiscreteTransitionMessagePassingRules))
    @test isempty(check_rule_ambiguities(DiscreteTransitionMessagePassingRules))
end
