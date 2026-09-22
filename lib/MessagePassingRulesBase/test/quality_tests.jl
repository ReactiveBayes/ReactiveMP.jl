@testitem "quality:aqua" tags = [:quality] begin
    using Aqua, MessagePassingRulesBase
    Aqua.test_all(MessagePassingRulesBase; deps_compat = (; check_extras = true))
end

@testitem "quality:closure" tags = [:quality] begin
    import Pkg

    # Checked on the resolved graph and on what a fresh process loads: each can pass while
    # the other fails.
    deps = Pkg.dependencies()
    self = only(filter(((_, info),) -> info.name == "MessagePassingRulesBase", deps))
    closure = Set{String}()
    frontier = collect(values(last(self).dependencies))
    while !isempty(frontier)
        info = get(deps, pop!(frontier), nothing)
        (info === nothing || info.name in closure) && continue
        push!(closure, info.name)
        append!(frontier, values(info.dependencies))
    end
    @test "BayesBase" in closure
    @test !("ExponentialFamily" in closure)

    probe = """
        using MessagePassingRulesBase
        loaded = Set(string(id.name) for id in keys(Base.loaded_modules))
        print("BayesBase" in loaded, " ", "ExponentialFamily" in loaded)
    """
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $probe`
    @test readchomp(cmd) == "true false"
end

@testitem "quality:doctests" tags = [:quality] begin
    using Documenter, MessagePassingRulesBase
    DocMeta.setdocmeta!(MessagePassingRulesBase, :DocTestSetup, :(using MessagePassingRulesBase); recursive = true)
    doctest(MessagePassingRulesBase; manual = false)
end
