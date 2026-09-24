@testitem "tables:passing" tags = [:testutils] setup = [ToyRules, Recording] begin
    using MessagePassingRulesTestUtils, Distributions, BayesBase
    T = ToyRules

    set = Recording.recorded() do
        @test_message_update_rule(
            node = T.Gauss, target = :out,
            cases = [
                (m = (μ = PointMass(1.0), σ = PointMass(2.0)),) => Normal(1.0, 2.0),
                (m = (μ = PointMass(-1.0), σ = PointMass(0.5)),) => Normal(-1.0, 0.5),
            ],
        )
        @test_message_update_rule(
            node = T.Gauss, target = :μ,
            cases = [(m = (out = Normal(0.0, 3.0), σ = PointMass(4.0)),) => ExpectedWithAnnotations(Normal(0.0, 5.0); logscale = 0.0)],
        )
        @test_marginal_update_rule(
            node = T.Gauss, target = (:out, :μ),
            cases = [(m = (out = Normal(1.0, 1.0), μ = Normal(2.0, 1.0)), q = (σ = PointMass(1.0),)) => (1.0, 2.0)],
        )
        @test_average_energy(
            node = T.Gauss,
            cases = [(q = (out = Normal(0.0, 1.0), μ = Normal(1.0, 1.0), σ = PointMass(1.0)),) => 1.5],
        )
        @test_message_update_rule(node = T.Buffered, target = :out, check_nonallocating = true, check_type_promotion = false, cases = [(m = (x = [1.0, 2.0],),) => [2.0, 4.0]])
    end
    @test isempty(Recording.failures(set))
    @test Recording.passes(set) > 20
end

@testitem "tables:type-promotion" tags = [:testutils] setup = [ToyRules, Recording] begin
    using MessagePassingRulesTestUtils, Distributions, BayesBase
    T = ToyRules
    # Default: all inputs converted, then each alone, for each of three float types.
    default = Recording.recorded() do
        @test_message_update_rule(node = T.Gauss, target = :out, cases = [(m = (μ = PointMass(1.0), σ = PointMass(2.0)),) => Normal(1.0, 2.0)])
    end
    exhaustive = Recording.recorded() do
        @test_message_update_rule(node = T.Gauss, target = :out, check_type_promotion = :exhaustive, cases = [(m = (μ = PointMass(1.0), σ = PointMass(2.0)),) => Normal(1.0, 2.0)])
    end
    @test isempty(Recording.failures(default)) && isempty(Recording.failures(exhaustive))
    @test Recording.passes(default) == 2 + 3 * 3
    @test Recording.passes(exhaustive) == 2 + 3 * 3
    # A rule whose output ignores the input type fails under promotion, and only there.
    widening = Recording.recorded() do
        @test_message_update_rule(node = T.Faulty, target = :out, cases = [(m = (x = PointMass(1.0),),) => PointMass(1.0)])
    end
    @test length(Recording.failures(widening)) == 2      # Float32 and BigFloat
    @test contains(Recording.failure_text(widening), "converted to Float32")
end

@testitem "tables:failures" tags = [:testutils] setup = [ToyRules, Recording] begin
    using MessagePassingRulesTestUtils, Distributions, BayesBase
    T = ToyRules

    wrong_value = Recording.recorded() do
        @test_message_update_rule(node = T.Gauss, target = :out, check_type_promotion = false, cases = [(m = (μ = PointMass(1.0), σ = PointMass(2.0)),) => Normal(1.0, 3.0)])
    end
    @test length(Recording.failures(wrong_value)) == 1
    @test contains(Recording.failure_text(wrong_value), "expected Distributions.Normal")
    # The failure points at the table's line in this file, not inside the package.
    @test contains(Recording.failure_text(wrong_value), "table_tests.jl")

    wrong_type = Recording.recorded() do
        @test_message_update_rule(node = T.Faulty, target = :x, check_type_promotion = false, cases = [(m = (out = PointMass(1.0),),) => PointMass(1.0)])
    end
    @test length(Recording.failures(wrong_type)) == 1

    wrong_annotation = Recording.recorded() do
        @test_message_update_rule(node = T.Gauss, target = :μ, check_type_promotion = false, cases = [(m = (out = Normal(0.0, 3.0), σ = PointMass(4.0)),) => ExpectedWithAnnotations(Normal(0.0, 5.0); logscale = 1.0)])
    end
    @test length(Recording.failures(wrong_annotation)) == 1

    ignores_buffer = Recording.recorded() do
        @test_message_update_rule(node = T.Buffered, target = :x, check_nonallocating = true, check_type_promotion = false, cases = [(m = (out = [2.0, 4.0],),) => [1.0, 2.0]])
    end
    text = Recording.failure_text(ignores_buffer)
    @test length(Recording.failures(ignores_buffer)) == 2
    @test contains(text, "instead of writing into its buffer") && contains(text, "allocated")

    missing_rule = Recording.recorded() do
        @test_message_update_rule(node = T.Gauss, target = :σ, cases = [(m = (out = Normal(0.0, 1.0),),) => Normal(0.0, 1.0)])
    end
    @test length(Recording.failures(missing_rule)) == 1
    @test contains(Recording.failure_text(missing_rule), "no message rule")

    @test_throws ArgumentError test_message_update_rule(T.Gauss, :out; cases = [(mm = (μ = 1,),) => 1])
end

@testitem "tables:factorized-cluster" tags = [:testutils] setup = [ToyRules, Recording] begin
    using MessagePassingRulesTestUtils, MessagePassingRulesBase, Distributions, BayesBase
    T = ToyRules
    inputs = (m = (out = Normal(1.0, 1.0), μ = Normal(2.0, 1.0), σ = PointMass(3.0)),)

    set = Recording.recorded() do
        @test_marginal_update_rule(
            node = T.Gauss, target = (:out, :μ, :σ),
            cases = [inputs => FactorizedCluster((:out, :μ) => PointMass([1.0, 2.0]), (:σ,) => PointMass(3.0))],
        )
    end
    @test isempty(Recording.failures(set))

    # Other blocks are a different result, even with the same numbers.
    set = Recording.recorded() do
        @test_marginal_update_rule(
            node = T.Gauss, target = (:out, :μ, :σ), check_type_promotion = false,
            cases = [inputs => FactorizedCluster((:out,) => PointMass(1.0), (:μ, :σ) => PointMass([2.0, 3.0]))],
        )
    end
    @test !isempty(Recording.failures(set))
end

@testitem "tables:struct-outputs" tags = [:testutils] begin
    using MessagePassingRulesTestUtils: approximately_equal

    # An output that is neither a number, an array nor a distribution, such as
    # ExponentialFamily's `JointNormal`, is compared field by field, within the tolerance.
    struct Joint{D, S}
        dist::D
        sizes::S
    end
    @test approximately_equal(Joint([1.0, 2.0], ((), ())), Joint([1.0 + 1.0e-12, 2.0], ((), ())); atol = 1.0e-10, rtol = 0.0)
    @test !approximately_equal(Joint([1.0, 2.0], ((), ())), Joint([1.1, 2.0], ((), ())); atol = 1.0e-10, rtol = 0.0)
    @test !approximately_equal(Joint([1.0], ((),)), Joint([1.0], ((1,),)); atol = 1.0e-10, rtol = 0.0)

    # A mutable struct is compared as `==` would, since its identity may matter.
    mutable struct Box
        x::Float64
    end
    box = Box(1.0)
    @test approximately_equal(box, box; atol = 1.0e-10, rtol = 0.0)
    @test !approximately_equal(Box(1.0), Box(1.0); atol = 1.0e-10, rtol = 0.0)
end

@testitem "tables:algorithm-extension" tags = [:testutils] setup = [ToyRules, Recording] begin
    using MessagePassingRulesTestUtils, Distributions, BayesBase
    T = ToyRules
    # A table run under an extension executes an inherited default rule as the engine does,
    # with `DefaultAlgorithm()`, through the rule, its preallocation and `rule!`.
    set = Recording.recorded() do
        @test_message_update_rule(node = T.Inherited, target = :out, algorithm = T.Extended(), cases = [(m = (x = PointMass(1.5),),) => PointMass(3.0)])
        @test_message_update_rule(
            node = T.Inherited, target = :x, algorithm = T.Extended(), check_nonallocating = true, check_type_promotion = false,
            cases = [(m = (out = [2.0, 4.0],),) => [1.0, 2.0]],
        )
    end
    @test isempty(Recording.failures(set))
    @test Recording.passes(set) > 5
end
