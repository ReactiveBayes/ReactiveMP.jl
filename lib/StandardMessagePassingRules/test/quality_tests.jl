@testitem "quality:aqua" tags = [:quality] begin
    using Aqua, StandardMessagePassingRules
    Aqua.test_all(StandardMessagePassingRules)
end

@testitem "quality:closure" tags = [:quality] begin
    import Pkg

    deps = Pkg.dependencies()
    self = only(filter(((_, info),) -> info.name == "StandardMessagePassingRules", deps))
    closure = Set{String}()
    frontier = collect(values(last(self).dependencies))
    while !isempty(frontier)
        info = get(deps, pop!(frontier), nothing)
        (info === nothing || info.name in closure) && continue
        push!(closure, info.name)
        append!(frontier, values(info.dependencies))
    end
    @test "MessagePassingRulesBase" in closure
    @test !("ReactiveMP" in closure)
    # The test tooling is for tests only, never a dependency of the rules.
    @test !("MessagePassingRulesTestUtils" in closure)
end

@testitem "quality:doctests" tags = [:quality] begin
    using Documenter, StandardMessagePassingRules
    DocMeta.setdocmeta!(StandardMessagePassingRules, :DocTestSetup, :(using StandardMessagePassingRules); recursive = true)
    doctest(StandardMessagePassingRules; manual = false)
end
