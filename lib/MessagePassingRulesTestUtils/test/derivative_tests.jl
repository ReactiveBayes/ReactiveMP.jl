@testitem "derivatives:propagate" tags = [:testutils] setup = [ToyRules, Recording] begin
    using MessagePassingRulesTestUtils, Distributions, BayesBase
    T = ToyRules
    set = Recording.recorded() do
        # Scalar θ through an allocating rule; the output's mean and scale both depend on it.
        @test_rule_derivatives(node = T.Gauss, target = :μ, inputs = θ -> (m = (out = Normal(θ, 2.0), σ = PointMass(θ^2)),), at = 1.5, summary = d -> mean(d) + std(d))
        # A vector θ through an in-place rule: checked through `rule` and through `rule!`.
        @test_rule_derivatives(node = T.Scaled, target = :out, inputs = θ -> (m = (x = [θ[1], θ[1] * θ[2]],),), at = [0.5, 3.0], summary = sum)
        # Under an extension, an inherited default rule runs with `DefaultAlgorithm()`.
        @test_rule_derivatives(node = T.Inherited, target = :x, algorithm = T.Extended(), inputs = θ -> (m = (out = [θ, 2θ],),), at = 1.0, summary = sum)
    end
    @test isempty(Recording.failures(set))
    @test Recording.passes(set) == 5
end

@testitem "derivatives:negative-control" tags = [:testutils] setup = [Recording] begin
    using MessagePassingRulesTestUtils, MessagePassingRulesBase, Distributions, BayesBase, ForwardDiff

    module Detached
    using MessagePassingRulesBase, BayesBase, ForwardDiff
    struct Node end
    @define_factor_node(node = Node, type = Stochastic, interfaces = [:out, :x])
    # Strips the dual part: the value is right, the derivative is silently zero.
    @define_message_update_rule(node = Node, target = :out, args = (m[:x]::PointMass,), body = (args) -> PointMass(2 * ForwardDiff.value(mean(args.m[:x]))))
    end

    set = Recording.recorded() do
        @test_rule_derivatives(node = Detached.Node, target = :out, inputs = θ -> (m = (x = PointMass(θ),),), at = 1.0)
    end
    @test length(Recording.failures(set)) == 1
    @test contains(Recording.failure_text(set), "ForwardDiff gives 0.0")
end
