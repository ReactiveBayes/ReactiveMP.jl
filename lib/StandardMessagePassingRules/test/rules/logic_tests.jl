# The logic nodes, against tables of cases. A marginal is the Contingency
# table of the inputs, rows `in1` and columns `in2`, false first.

@testitem "rules:AND" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions

    @test_message_update_rule(node = AND, target = :out, cases = [(m = (in1 = Bernoulli(0.3), in2 = Bernoulli(0.5)),) => Bernoulli(0.15)])
    @test_message_update_rule(node = AND, target = :in1, cases = [(m = (out = Bernoulli(0.3), in2 = Bernoulli(0.4)),) => Bernoulli(0.54 / 1.24)])
    @test_message_update_rule(node = AND, target = :in2, cases = [(m = (out = Bernoulli(0.6), in1 = Bernoulli(0.5)),) => Bernoulli(0.5 / 0.9)])
    @test_marginal_update_rule(
        node = AND, target = (:in1, :in2),
        cases = [(m = (out = Bernoulli(0.2), in1 = Bernoulli(0.8), in2 = Bernoulli(0.4)),) => Contingency([0.2 * 0.8 * 0.6 0.2 * 0.8 * 0.4; 0.8 * 0.8 * 0.6 0.2 * 0.8 * 0.4])],
    )
end

@testitem "rules:OR" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions

    @test_message_update_rule(node = OR, target = :out, cases = [(m = (in1 = Bernoulli(0.3), in2 = Bernoulli(0.4)),) => Bernoulli(0.58)])
    @test_message_update_rule(node = OR, target = :in1, cases = [(m = (out = Bernoulli(0.3), in2 = Bernoulli(0.4)),) => Bernoulli(0.3 / 0.84)])
    @test_message_update_rule(node = OR, target = :in2, cases = [(m = (out = Bernoulli(0.6), in1 = Bernoulli(0.5)),) => Bernoulli(0.6 / 1.1)])
    @test_marginal_update_rule(
        node = OR, target = (:in1, :in2),
        cases = [(m = (out = Bernoulli(0.2), in1 = Bernoulli(0.8), in2 = Bernoulli(0.4)),) => Contingency([0.8 * 0.2 * 0.6 0.2 * 0.4 * 0.2; 0.8 * 0.2 * 0.6 0.2 * 0.8 * 0.4])],
    )
end

@testitem "rules:NOT" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions

    @test_message_update_rule(node = NOT, target = :out, cases = [(m = (in = Bernoulli(0.3),),) => Bernoulli(0.7)])
    @test_message_update_rule(node = NOT, target = :in, cases = [(m = (out = Bernoulli(0.6),),) => Bernoulli(0.4)])
end

@testitem "rules:IMPLY" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions

    @test_message_update_rule(node = IMPLY, target = :out, cases = [(m = (in1 = Bernoulli(0.4), in2 = Bernoulli(0.7)),) => Bernoulli(0.88)])
    @test_message_update_rule(node = IMPLY, target = :in1, cases = [(m = (out = Bernoulli(0.2), in2 = Bernoulli(0.5)),) => Bernoulli(0.5 / 0.7)])
    @test_message_update_rule(node = IMPLY, target = :in2, cases = [(m = (out = Bernoulli(0.3), in1 = Bernoulli(0.4)),) => Bernoulli(0.3 / 0.76)])
    @test_marginal_update_rule(
        node = IMPLY, target = (:in1, :in2),
        cases = [(m = (out = Bernoulli(0.2), in1 = Bernoulli(0.8), in2 = Bernoulli(0.4)),) => Contingency([0.2 * 0.2 * 0.6 0.2 * 0.2 * 0.4; 0.8 * 0.8 * 0.6 0.2 * 0.8 * 0.4])],
    )
end
