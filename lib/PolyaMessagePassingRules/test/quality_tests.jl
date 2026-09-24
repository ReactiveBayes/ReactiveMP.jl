@testitem "quality:aqua" tags = [:quality] begin
    using Aqua, PolyaMessagePassingRules
    Aqua.test_all(PolyaMessagePassingRules)
end

@testitem "quality:closure" tags = [:quality] begin
    import Pkg

    deps = Pkg.dependencies()
    self = only(filter(((_, info),) -> info.name == "PolyaMessagePassingRules", deps))
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
    # The GPL-3 dependency is this package's alone.
    @test "PolyaGammaHybridSamplers" in closure
    @test !("ReactiveMP" in closure)
    # The test tooling is for tests only, never a dependency of the rules.
    @test !("MessagePassingRulesTestUtils" in closure)
end

@testitem "quality:doctests" tags = [:quality] begin
    using Documenter, PolyaMessagePassingRules
    DocMeta.setdocmeta!(PolyaMessagePassingRules, :DocTestSetup, :(using PolyaMessagePassingRules); recursive = true)
    doctest(PolyaMessagePassingRules; manual = false)
end

@testitem "quality:rules" tags = [:quality] begin
    using PolyaMessagePassingRules
    using MessagePassingRulesBase: check_rules, check_rule_ambiguities

    # Every rule agrees with its node's declaration and with its algorithm's declared
    # dependencies, and no call could match two rules equally well.
    @test isempty(check_rules(PolyaMessagePassingRules))
    @test isempty(check_rule_ambiguities(PolyaMessagePassingRules))
end

@testitem "quality:nodes" tags = [:quality] begin
    using PolyaMessagePassingRules, MessagePassingRulesBase
    using MessagePassingRulesBase: dependencies_spec, target_dependencies, extends_default_scheme, default_algorithm, Target

    @test default_algorithm(BinomialPolya) === BinomialPolyaApproximation(nothing)
    @test default_algorithm(MultinomialPolya) === MultinomialPolyaApproximation(21)
    # The rules towards the weights also read the message on their own edge, v6's
    # `RequireMessageFunctionalDependencies`; every other target follows the factorisation.
    for (node, algorithm, weights, others) in ((BinomialPolya, BinomialPolyaApproximation(samples = 10), :β, (:y, :x, :n)), (MultinomialPolya, MultinomialPolyaApproximation(), :ψ, (:x, :N)))
        spec = dependencies_spec(node, algorithm)
        @test map(d -> (d.container, d.key), target_dependencies(spec, Target(weights))) == ((:m, weights),)
        @test all(t -> extends_default_scheme(spec, Target(t)) && isempty(target_dependencies(spec, Target(t))), others)
    end
end
