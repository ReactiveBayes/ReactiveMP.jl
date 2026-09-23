@testitem "quality:aqua" tags = [:quality] begin
    using Aqua, MessagePassingRulesTestUtils
    Aqua.test_all(MessagePassingRulesTestUtils; deps_compat = (; check_extras = true))
end

@testitem "quality:closure" tags = [:quality] begin
    import Pkg

    deps = Pkg.dependencies()
    self = only(filter(((_, info),) -> info.name == "MessagePassingRulesTestUtils", deps))
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
    @test !("ExponentialFamily" in closure)
end

@testitem "quality:doctests" tags = [:quality] begin
    using Documenter, MessagePassingRulesTestUtils
    DocMeta.setdocmeta!(MessagePassingRulesTestUtils, :DocTestSetup, :(using MessagePassingRulesTestUtils); recursive = true)
    doctest(MessagePassingRulesTestUtils; manual = false)
end
