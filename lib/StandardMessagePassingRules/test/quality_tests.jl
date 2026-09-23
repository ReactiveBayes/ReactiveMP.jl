@testitem "quality:aqua" tags = [:quality] begin
    using Aqua, StandardMessagePassingRules, ExponentialFamily
    # A rule package declares nodes, marginal rules and average energies for distributions
    # another package owns, through functions the base package owns. That is the design, and
    # Aqua reports it as piracy, so the node types are declared as owned here. Message rules
    # never show up: their target's `Symbol` parameter makes them look owned to Aqua.
    # An alias such as `Categorical`, a `DiscreteNonParametric` with fixed parameters, is
    # compared by its underlying type. The same list covers the Uniform(0, 1)×Beta product,
    # which is defined here for two types this package does not own (an upstream candidate).
    # So does `public_equivalent` for ExponentialFamily's fast Wishart types.
    owned = unique([StandardMessagePassingRules.NODES; map(T -> Base.unwrap_unionall(T).name.wrapper, StandardMessagePassingRules.NODES); ExponentialFamily.WishartFast; ExponentialFamily.InverseWishartFast])
    Aqua.test_all(StandardMessagePassingRules; piracies = (treat_as_own = owned,))
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

@testitem "quality:rules" tags = [:quality] begin
    using StandardMessagePassingRules
    using MessagePassingRulesBase: check_rules, check_rule_ambiguities

    # Every rule agrees with its node's declaration and with its algorithm's declared
    # dependencies, and no call could match two rules equally well.
    @test isempty(check_rules(StandardMessagePassingRules))
    @test isempty(check_rule_ambiguities(StandardMessagePassingRules))
end
