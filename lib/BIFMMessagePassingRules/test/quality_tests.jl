@testitem "quality:aqua" tags = [:quality] begin
    using Aqua, BIFMMessagePassingRules
    Aqua.test_all(BIFMMessagePassingRules)
end

@testitem "quality:closure" tags = [:quality] begin
    import Pkg

    deps = Pkg.dependencies()
    self = only(filter(((_, info),) -> info.name == "BIFMMessagePassingRules", deps))
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
    using Documenter, BIFMMessagePassingRules
    DocMeta.setdocmeta!(BIFMMessagePassingRules, :DocTestSetup, :(using BIFMMessagePassingRules); recursive = true)
    doctest(BIFMMessagePassingRules; manual = false)
end

@testitem "quality:rules" tags = [:quality] begin
    using BIFMMessagePassingRules
    using MessagePassingRulesBase: check_rules, check_rule_ambiguities

    # Every rule agrees with its node's declaration and with its algorithm's declared
    # dependencies, and no call could match two rules equally well.
    @test isempty(check_rules(BIFMMessagePassingRules))
    @test isempty(check_rule_ambiguities(BIFMMessagePassingRules))
end

@testitem "quality:nodes" tags = [:quality] begin
    using BIFMMessagePassingRules, MessagePassingRulesBase
    using MessagePassingRulesBase: dependencies_spec, target_dependencies, extends_default_scheme, default_algorithm, Target, ispure

    # BIFM has no algorithm of its own: the model gives BIFMSmoother, whose rules are pure.
    @test default_algorithm(BIFM) === DefaultAlgorithm()
    algorithm = BIFMSmoother([1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0], [1.0 0.0])
    spec = dependencies_spec(BIFM, algorithm)
    # The forward rules also read their own edge's message, where v6 read a cache.
    for (target, own) in ((:out, ((:m, :out),)), (:in, ((:m, :in),)), (:zprev, ()), (:znext, ((:m, :znext),)))
        @test extends_default_scheme(spec, Target(target))
        @test map(d -> (d.container, d.key), target_dependencies(spec, Target(target))) == own
    end
    @test ispure(typeof(algorithm))
    # BIFMHelper: towards `in` the message on `out`, towards `out` the factorisation's inputs.
    helper = dependencies_spec(BIFMHelper, DefaultAlgorithm())
    @test map(d -> (d.container, d.key), target_dependencies(helper, Target(:in))) == ((:m, :out),)
    @test extends_default_scheme(helper, Target(:out))
end
