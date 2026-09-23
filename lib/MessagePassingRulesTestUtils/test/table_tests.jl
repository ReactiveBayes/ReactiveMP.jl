@testitem "tables:passing" tags = [:testutils] setup = [ToyRules, Recording] begin
    using MessagePassingRulesTestUtils, Distributions, BayesBase
    T = ToyRules

    set = Recording.recorded() do
        @test_message_update_rule(
            node = T.Gauss, towards = :out,
            cases = [
                (m = (μ = PointMass(1.0), σ = PointMass(2.0)),) => Normal(1.0, 2.0),
                (m = (μ = PointMass(-1.0), σ = PointMass(0.5)),) => Normal(-1.0, 0.5),
            ],
        )
        @test_message_update_rule(
            node = T.Gauss, towards = :μ,
            cases = [(m = (out = Normal(0.0, 3.0), σ = PointMass(4.0)),) => ExpectedWithAnnotations(Normal(0.0, 5.0); logscale = 0.0)],
        )
        @test_marginal_update_rule(
            node = T.Gauss, towards = (:out, :μ),
            cases = [(m = (out = Normal(1.0, 1.0), μ = Normal(2.0, 1.0)), q = (σ = PointMass(1.0),)) => (1.0, 2.0)],
        )
        @test_average_energy(
            node = T.Gauss,
            cases = [(q = (out = Normal(0.0, 1.0), μ = Normal(1.0, 1.0), σ = PointMass(1.0)),) => 1.5],
        )
        @test_message_update_rule(node = T.Buffered, towards = :out, check_nonallocating = true, check_type_promotion = false, cases = [(m = (x = [1.0, 2.0],),) => [2.0, 4.0]])
    end
    @test isempty(Recording.failures(set))
    @test Recording.passes(set) > 20
end

@testitem "tables:type-promotion" tags = [:testutils] setup = [ToyRules, Recording] begin
    using MessagePassingRulesTestUtils, Distributions, BayesBase
    T = ToyRules
    # Default: all inputs converted, then each alone, for each of three float types.
    default = Recording.recorded() do
        @test_message_update_rule(node = T.Gauss, towards = :out, cases = [(m = (μ = PointMass(1.0), σ = PointMass(2.0)),) => Normal(1.0, 2.0)])
    end
    exhaustive = Recording.recorded() do
        @test_message_update_rule(node = T.Gauss, towards = :out, check_type_promotion = :exhaustive, cases = [(m = (μ = PointMass(1.0), σ = PointMass(2.0)),) => Normal(1.0, 2.0)])
    end
    @test isempty(Recording.failures(default)) && isempty(Recording.failures(exhaustive))
    @test Recording.passes(default) == 2 + 3 * 3
    @test Recording.passes(exhaustive) == 2 + 3 * 3
    # A rule whose output ignores the input type fails under promotion, and only there.
    widening = Recording.recorded() do
        @test_message_update_rule(node = T.Faulty, towards = :out, cases = [(m = (x = PointMass(1.0),),) => PointMass(1.0)])
    end
    @test length(Recording.failures(widening)) == 2      # Float32 and BigFloat
    @test contains(Recording.failure_text(widening), "converted to Float32")
end

@testitem "tables:failures" tags = [:testutils] setup = [ToyRules, Recording] begin
    using MessagePassingRulesTestUtils, Distributions, BayesBase
    T = ToyRules

    wrong_value = Recording.recorded() do
        @test_message_update_rule(node = T.Gauss, towards = :out, check_type_promotion = false, cases = [(m = (μ = PointMass(1.0), σ = PointMass(2.0)),) => Normal(1.0, 3.0)])
    end
    @test length(Recording.failures(wrong_value)) == 1
    @test contains(Recording.failure_text(wrong_value), "expected Distributions.Normal")
    # The failure points at the table's line in this file, not inside the package.
    @test contains(Recording.failure_text(wrong_value), "table_tests.jl")

    wrong_type = Recording.recorded() do
        @test_message_update_rule(node = T.Faulty, towards = :x, check_type_promotion = false, cases = [(m = (out = PointMass(1.0),),) => PointMass(1.0)])
    end
    @test length(Recording.failures(wrong_type)) == 1

    wrong_annotation = Recording.recorded() do
        @test_message_update_rule(node = T.Gauss, towards = :μ, check_type_promotion = false, cases = [(m = (out = Normal(0.0, 3.0), σ = PointMass(4.0)),) => ExpectedWithAnnotations(Normal(0.0, 5.0); logscale = 1.0)])
    end
    @test length(Recording.failures(wrong_annotation)) == 1

    ignores_buffer = Recording.recorded() do
        @test_message_update_rule(node = T.Buffered, towards = :x, check_nonallocating = true, check_type_promotion = false, cases = [(m = (out = [2.0, 4.0],),) => [1.0, 2.0]])
    end
    text = Recording.failure_text(ignores_buffer)
    @test length(Recording.failures(ignores_buffer)) == 2
    @test contains(text, "instead of writing into its buffer") && contains(text, "allocated")

    missing_rule = Recording.recorded() do
        @test_message_update_rule(node = T.Gauss, towards = :σ, cases = [(m = (out = Normal(0.0, 1.0),),) => Normal(0.0, 1.0)])
    end
    @test length(Recording.failures(missing_rule)) == 1
    @test contains(Recording.failure_text(missing_rule), "no message rule")

    @test_throws ArgumentError test_message_update_rule(T.Gauss, :out; cases = [(mm = (μ = 1,),) => 1])
end
