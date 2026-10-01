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

@testitem "derivatives:a missing rule, or one that cannot take dual numbers" tags = [:testutils] setup = [ToyRules, Recording] begin
    using MessagePassingRulesTestUtils, MessagePassingRulesBase, Distributions, BayesBase, Test

    module Typed
    using MessagePassingRulesBase, BayesBase
    struct Node end
    @define_factor_node(node = Node, type = Stochastic, interfaces = [:out, :x])
    # Writes into a Float64 buffer: a dual number cannot be stored there.
    @define_message_update_rule(node = Node, target = :out, args = (m[:x]::PointMass,), body = (args) -> (buffer = zeros(1); buffer[1] = 2 * mean(args.m[:x]); PointMass(buffer[1])))
    end

    set = Recording.recorded() do
        # No rule towards `x`: a failure, with the near misses.
        @test_rule_derivatives(node = Typed.Node, target = :x, inputs = θ -> (m = (out = PointMass(θ),),), at = 1.0)
        # ForwardDiff fails inside the rule; the central difference runs, and no comparison is made.
        @test_rule_derivatives(node = Typed.Node, target = :out, inputs = θ -> (m = (x = PointMass(θ),),), at = 1.0)
        # The checks after them still run.
        @test_rule_derivatives(node = ToyRules.Gauss, target = :μ, inputs = θ -> (m = (out = Normal(θ, 2.0), σ = PointMass(θ^2)),), at = 1.5)
    end
    @test count(r -> r isa Test.Error, set.results) == 0
    @test length(Recording.failures(set)) == 2 && Recording.passes(set) == 1
    text = Recording.failure_text(set)
    @test contains(text, "rule_found") && contains(text, "the rule for $(Typed.Node) target :x at 1.0")
    @test contains(text, "rule_runs") && contains(text, "by ForwardDiff: the rule threw") && !contains(text, "by a central difference")
end
