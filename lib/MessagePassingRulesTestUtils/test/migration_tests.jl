@testitem "migration:compare" tags = [:testutils] setup = [Recording] begin
    using MessagePassingRulesTestUtils, Distributions

    declared = [DeclaredDisagreement("NMV:μ:vmp"; kind = :correction, reasoning = "the reference uses E[v] where naive VMP needs 1/E[1/v]")]
    records = MigrationRecord[]
    set = Recording.recorded() do
        push!(records, compare_with_reference("agree", Normal(0.0, 1.0), Normal(0.0, 1.0); actual_logscale = 0.0, reference_logscale = 0.0))
        push!(records, compare_with_reference("NMV:μ:vmp", Normal(0.0, 1.0), Normal(0.0, 2.0); declared))
        push!(records, compare_with_reference("undeclared", Normal(0.0, 1.0), Normal(0.0, 2.0); declared))
        push!(records, compare_with_reference("logscale", Normal(0.0, 1.0), Normal(0.0, 1.0); actual_logscale = 0.0, reference_logscale = -1.0))
    end
    @test map(r -> r.outcome, records) == [:agree, :correction, :disagree, :disagree]
    @test length(Recording.failures(set)) == 2
    @test contains(Recording.failure_text(set), "declare it a :migration_bug or a :correction")

    @test_throws ArgumentError DeclaredDisagreement("x"; kind = :whatever, reasoning = "r")
    @test_throws ArgumentError DeclaredDisagreement("x"; kind = :correction, reasoning = "  ")
end

@testitem "migration:fixtures" tags = [:testutils] begin
    using MessagePassingRulesTestUtils, Distributions, Serialization
    record = MigrationRecord("id", "Node", ":out", (m = (μ = 1.0,),), Normal(0.0, 1.0), Normal(0.0, 1.0), 0.0, 0.0, :agree)
    path = joinpath(mktempdir(), "fixtures.jls")
    save_migration_fixtures(path, [record]; packages = Dict("ReactiveMP" => v"6.5.0"))
    loaded = load_migration_fixtures(path)
    @test only(loaded.records).actual == Normal(0.0, 1.0)
    @test loaded.header.packages["ReactiveMP"] == v"6.5.0"
    @test loaded.header.julia == VERSION

    # A file written by another Julia minor is refused rather than half-read.
    header = (format = 1, julia = VersionNumber(VERSION.major, VERSION.minor + 1), packages = Dict{String, Any}())
    open(io -> serialize(io, (header, [record])), path, "w")
    @test_throws ArgumentError load_migration_fixtures(path)
end
