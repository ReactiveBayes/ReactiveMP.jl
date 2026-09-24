@testitem "engine-fixtures:encoding" tags = [:testutils] begin
    using MessagePassingRulesTestUtils, MessagePassingRulesBase, Distributions, BayesBase

    @test encode_fixture_value(1) == 1.0
    @test encode_fixture_value(Normal(0.5, 2.0)) == Dict("type" => "Normal", "params" => [0.5, 2.0])
    @test encode_fixture_value(PointMass(3.0)) == Dict("type" => "PointMass", "params" => [3.0])
    @test encode_fixture_value(MixtureDistribution([Normal(0.0, 1.0), Normal(2.0, 0.5)], [0.25, 0.75])) == Dict(
        "type" => "MixtureDistribution", "components" => Any[Dict("type" => "Normal", "params" => [0.0, 1.0]), Dict("type" => "Normal", "params" => [2.0, 0.5])], "weights" => [0.25, 0.75],
    )
    @test encode_fixture_value(Dirichlet([1.0, 2.0])) == Dict("type" => "Dirichlet", "params" => [[1.0, 2.0]])
    # Matrices are stored row by row, so a covariance survives a text round trip.
    @test encode_fixture_value(MvNormal([0.0, 1.0], [2.0 0.5; 0.5 1.0]))["params"] == [[0.0, 1.0], [[2.0, 0.5], [0.5, 1.0]]]
    @test encode_fixture_value((Normal(0.0, 1.0), PointMass(1.0))) == [encode_fixture_value(Normal(0.0, 1.0)), encode_fixture_value(PointMass(1.0))]
    @test encode_fixture_value(nothing) == Dict("type" => "nothing")
    @test encode_fixture_value(MessagePassingRulesBase.FactorizedCluster((:out, :μ) => PointMass([1.0, 2.0]), (:v,) => PointMass(3.0))) ==
        Dict("type" => "FactorizedCluster", "blocks" => [["out", "μ"], ["v"]], "components" => [encode_fixture_value(PointMass([1.0, 2.0])), encode_fixture_value(PointMass(3.0))])
    # A block may hold a member of a group, and an array of more axes is stored with its size.
    @test encode_fixture_value(MessagePassingRulesBase.FactorizedCluster((:out,) => PointMass(1.0), (:in, (:T, 1)) => PointMass(2.0)))["blocks"] == [["out"], ["in", "(:T, 1)"]]
    @test encode_fixture_value(reshape(collect(1.0:8.0), 2, 2, 2)) == Dict("type" => "Array", "size" => [2, 2, 2], "values" => collect(1.0:8.0))
    # A rule call skipped for a missing input records `missing`, which is not `nothing`.
    @test encode_fixture_value(missing) == Dict("type" => "missing")
end

@testitem "engine-fixtures:round-trip" tags = [:testutils] begin
    using MessagePassingRulesTestUtils, Distributions, BayesBase

    trajectory = EngineTrajectory(
        "toy";
        description = "x ~ N(0, 1), one observation",
        free_energy = [1.5, 1.25],
        posteriors = Dict("x" => Normal(0.5, 0.7), "m" => [Normal(-1.0, 1.0), Normal(1.0, 1.0)]),
        trace = [
            RuleCallRecord(1, "Normal", ":out", Normal(0.0, 1.0), 0.0),
            RuleCallRecord(1, "Normal", ":μ", Normal(1.0, 1.0), nothing),
            RuleCallRecord(2, "Normal", "(:m, 2)", PointMass(1.0), -0.5),
        ],
    )
    path = joinpath(mktempdir(), "toy.toml")
    save_engine_fixture(path, trajectory; packages = Dict("ReactiveMP" => v"6.5.0"), notes = "recorded by a test")
    header, loaded = load_engine_fixture(path)

    @test header.format == 1
    @test header.packages["ReactiveMP"] == "6.5.0"
    @test header.notes == "recorded by a test"
    @test VersionNumber(header.julia) == VERSION
    @test loaded.id == "toy"
    @test loaded.free_energy == [1.5, 1.25]
    @test loaded.posteriors["m"] == encode_fixture_value([Normal(-1.0, 1.0), Normal(1.0, 1.0)])
    @test [r.target for r in loaded.trace] == [":out", ":μ", "(:m, 2)"]
    @test [r.logscale for r in loaded.trace] == [0.0, nothing, -0.5]
    @test loaded.trace[3].result == encode_fixture_value(PointMass(1.0))

    # A fixture is text, readable on any Julia version: the header is plain TOML.
    @test contains(read(path, String), "format = 1")
end

@testitem "engine-fixtures:compare" tags = [:testutils] setup = [Recording] begin
    using MessagePassingRulesTestUtils, Distributions

    make(; fe = [2.0, 1.0], x = Normal(0.5, 0.7), trace = [RuleCallRecord(1, "Normal", ":out", Normal(0.0, 1.0), 0.0)]) =
        EngineTrajectory("toy"; free_energy = fe, posteriors = Dict("x" => x), trace = trace)
    reference = make()

    set = Recording.recorded() do
        @test compare_engine_trajectory(make(), reference) === :agree
    end
    @test isempty(Recording.failures(set))

    # Each part is checked and reported on its own.
    set = Recording.recorded() do
        @test compare_engine_trajectory(make(fe = [2.0, 1.1]), reference) === :disagree
    end
    @test length(Recording.failures(set)) == 1
    @test contains(Recording.failure_text(set), "free energy")

    set = Recording.recorded() do
        compare_engine_trajectory(make(x = Normal(0.5, 0.8)), reference)
    end
    @test contains(Recording.failure_text(set), "posterior `x`")

    # Order matters: the same calls in another order are a disagreement.
    two = [RuleCallRecord(1, "Normal", ":out", Normal(0.0, 1.0), 0.0), RuleCallRecord(1, "Normal", ":μ", Normal(1.0, 1.0), 0.0)]
    set = Recording.recorded() do
        compare_engine_trajectory(make(trace = reverse(two)), make(trace = two))
    end
    @test contains(Recording.failure_text(set), "rule call 1")

    # Unless the order within an iteration is declared free: then the same calls with the
    # same results agree in any order, and a missing, extra or different call still does not.
    set = Recording.recorded() do
        @test compare_engine_trajectory(make(trace = reverse(two)), make(trace = two); trace_order = :within_iteration) === :agree
    end
    @test isempty(Recording.failures(set))
    moved = [RuleCallRecord(2, "Normal", ":out", Normal(0.0, 1.0), 0.0), two[2]]
    set = Recording.recorded() do
        compare_engine_trajectory(make(trace = moved), make(trace = two); trace_order = :within_iteration)
    end
    @test contains(Recording.failure_text(set), "iteration 1")
    changed = [two[1], RuleCallRecord(1, "Normal", ":μ", Normal(1.0, 2.0), 0.0)]
    set = Recording.recorded() do
        compare_engine_trajectory(make(trace = changed), make(trace = two); trace_order = :within_iteration)
    end
    @test contains(Recording.failure_text(set), "Normal(:μ)")
    @test_throws ArgumentError compare_engine_trajectory(reference, reference; trace_order = :none)

    # v6 computes a message once per subscriber; declared, a v6 call repeating an earlier one of
    # its iteration, with the same result, is dropped. A repeat with another result is not.
    repeated = [two[1], two[1], two[2]]
    set = Recording.recorded() do
        compare_engine_trajectory(make(trace = two), make(trace = repeated))
    end
    @test !isempty(Recording.failures(set))
    set = Recording.recorded() do
        @test compare_engine_trajectory(make(trace = two), make(trace = repeated); collapse_repeats = true) === :agree
    end
    @test isempty(Recording.failures(set))
    differing = [two[1], RuleCallRecord(1, "Normal", ":out", Normal(0.0, 2.0), 0.0), two[2]]
    set = Recording.recorded() do
        compare_engine_trajectory(make(trace = two), make(trace = differing); collapse_repeats = true)
    end
    @test !isempty(Recording.failures(set))
    # A repeat in another iteration is a call of its own.
    later = [two[1], two[2], RuleCallRecord(2, "Normal", ":out", Normal(0.0, 1.0), 0.0)]
    set = Recording.recorded() do
        compare_engine_trajectory(make(trace = two), make(trace = later); collapse_repeats = true)
    end
    @test !isempty(Recording.failures(set))

    set = Recording.recorded() do
        compare_engine_trajectory(make(trace = [RuleCallRecord(1, "Normal", ":out", Normal(0.0, 1.0), -1.0)]), reference)
    end
    @test contains(Recording.failure_text(set), "log scale")

    # A declared disagreement is reported with its kind and does not fail.
    declared = [DeclaredDisagreement("toy"; kind = :correction, reasoning = "v6 is wrong here")]
    set = Recording.recorded() do
        @test compare_engine_trajectory(make(fe = [2.0, 1.1]), reference; declared) === :correction
    end
    @test isempty(Recording.failures(set))

    # A value that was recorded from v7 is encoded before comparison, so the two sides
    # may be given either as distributions or as their encodings.
    encoded = EngineTrajectory("toy"; free_energy = [2.0, 1.0], posteriors = Dict("x" => encode_fixture_value(Normal(0.5, 0.7))), trace = reference.trace)
    set = Recording.recorded() do
        @test compare_engine_trajectory(encoded, reference) === :agree
    end
    @test isempty(Recording.failures(set))
end

@testitem "engine-fixtures:terminal prod argument" tags = [:testutils] begin
    using MessagePassingRulesTestUtils, BayesBase, Distributions
    using BayesBase: TerminalProdArgument

    # A marginal sent as a message encodes as its argument, wrapped, and encodes to itself.
    encoded = encode_fixture_value(TerminalProdArgument(Normal(1.0, 2.0)))
    @test encoded == Dict{String, Any}("type" => "TerminalProdArgument", "argument" => Dict{String, Any}("type" => "Normal", "params" => Any[1.0, 2.0]))
    @test encode_fixture_value(encoded) == encoded
end
