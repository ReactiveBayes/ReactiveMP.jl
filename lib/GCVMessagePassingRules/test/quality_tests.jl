@testitem "quality:aqua" tags = [:quality] begin
    using Aqua, GCVMessagePassingRules
    Aqua.test_all(GCVMessagePassingRules)
end

@testitem "quality:closure" tags = [:quality] begin
    import Pkg

    deps = Pkg.dependencies()
    self = only(filter(((_, info),) -> info.name == "GCVMessagePassingRules", deps))
    closure = Set{String}()
    frontier = collect(values(last(self).dependencies))
    while !isempty(frontier)
        info = get(deps, pop!(frontier), nothing)
        (info === nothing || info.name in closure) && continue
        push!(closure, info.name)
        append!(frontier, values(info.dependencies))
    end
    @test "MessagePassingRulesBase" in closure
    @test "MessagePassingRulesApproximations" in closure
    @test !("ReactiveMP" in closure)
    # The test tooling is for tests only, never a dependency of the rules.
    @test !("MessagePassingRulesTestUtils" in closure)
end

@testitem "quality:doctests" tags = [:quality] begin
    using Documenter, GCVMessagePassingRules
    DocMeta.setdocmeta!(GCVMessagePassingRules, :DocTestSetup, :(using GCVMessagePassingRules); recursive = true)
    doctest(GCVMessagePassingRules; manual = false)
end

@testitem "quality:rules" tags = [:quality] begin
    using GCVMessagePassingRules
    using MessagePassingRulesBase: check_rules, check_rule_ambiguities

    # Every rule agrees with its node's declaration and with its algorithm's declared
    # dependencies, and no call could match two rules equally well.
    @test isempty(check_rules(GCVMessagePassingRules))
    @test isempty(check_rule_ambiguities(GCVMessagePassingRules))
end
