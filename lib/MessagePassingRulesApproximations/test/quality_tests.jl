@testitem "quality:aqua" tags = [:quality] begin
    using Aqua, MessagePassingRulesApproximations
    Aqua.test_all(MessagePassingRulesApproximations; deps_compat = (; check_extras = true))
end

@testitem "quality:closure" tags = [:quality] begin
    import Pkg

    deps = Pkg.dependencies()
    self = only(filter(((_, info),) -> info.name == "MessagePassingRulesApproximations", deps))
    closure = Set{String}()
    frontier = collect(values(last(self).dependencies))
    while !isempty(frontier)
        info = get(deps, pop!(frontier), nothing)
        (info === nothing || info.name in closure) && continue
        push!(closure, info.name)
        append!(frontier, values(info.dependencies))
    end
    # Numerics only: no rule system and no distribution package.
    @test !("MessagePassingRulesBase" in closure)
    @test !("ExponentialFamily" in closure)
    @test !("Distributions" in closure)
    @test !("ReactiveMP" in closure)
end

@testitem "quality:doctests" tags = [:quality] begin
    using Documenter, MessagePassingRulesApproximations
    DocMeta.setdocmeta!(MessagePassingRulesApproximations, :DocTestSetup, :(using MessagePassingRulesApproximations); recursive = true)
    doctest(MessagePassingRulesApproximations; manual = false)
end
