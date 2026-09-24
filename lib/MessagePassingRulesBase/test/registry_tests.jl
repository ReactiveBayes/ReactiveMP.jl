@testitem "registry:in-process" tags = [:base] begin
    using MessagePassingRulesBase: RuleSpec, RuleArgs, Target, DefaultAlgorithm, register!, registered_rules, registries,
        duplicate_rules, @define_registry

    module First
    using MessagePassingRulesBase: @define_registry
    struct Node end
    @define_registry
    @define_registry          # idempotent
    end

    body1 = (o, s, a, c, r, n, t) -> 1
    body2 = (o, s, a, c, r, n, t) -> 2
    spec(body) = RuleSpec(kind = :message, node = First.Node, target = Target{:out}, algorithm = DefaultAlgorithm, signature = RuleArgs, body = body)

    register!(First.__message_passing_registry__, spec(body1))
    ours() = filter(s -> s.node === First.Node, registered_rules())
    @test length(ours()) == 1

    # Redefinition with the same signature replaces the entry.
    register!(First.__message_passing_registry__, spec(body2))
    @test length(ours()) == 1
    @test only(ours()).body === body2
    @test any(((m, _),) -> m === First, registries())

    # The same signature from a second module is a duplicate, reported as such.
    module Second
    using MessagePassingRulesBase: @define_registry
    @define_registry
    end
    @test isempty(filter(g -> any(((m, s),) -> s.node === First.Node, g), duplicate_rules()))
    register!(Second.__message_passing_registry__, spec(body1))
    groups = filter(g -> any(((m, s),) -> s.node === First.Node, g), duplicate_rules())
    @test length(groups) == 1
    @test Set(first.(only(groups))) == Set([First, Second])
end

@testitem "registry:lifecycle" tags = [:base, :slow] begin
    import Pkg

    # Real packages, precompiled, then loaded by fresh processes: the only way to see what a
    # precompile image actually carries.
    base = pkgdir(MessagePassingRulesBase)
    fixtures = joinpath(base, "test", "fixtures")
    env = mktempdir()
    julia = Base.julia_cmd()
    setup = """
        import Pkg
        Pkg.activate($(repr(env)); io = devnull)
        Pkg.develop([Pkg.PackageSpec(path = p) for p in $(repr([base, joinpath(fixtures, "RegistryFixture"), joinpath(fixtures, "FixtureHost"), joinpath(fixtures, "FixtureWeak")]))]; io = devnull)
        Pkg.precompile(; io = devnull)
    """
    run(`$julia --startup-file=no -e $setup`)

    probe(code) = readchomp(`$julia --startup-file=no --project=$env -e $code`)
    count_for(node) = "count(s -> s.node === $node, MessagePassingRulesBase.registered_rules())"

    # Fresh load after precompilation, including a nested module.
    @test probe(
        """
            import MessagePassingRulesBase, RegistryFixture
            print(Base.isprecompiled(Base.PkgId(RegistryFixture)), " ",
                  $(count_for("RegistryFixture.Node")), " ", $(count_for("RegistryFixture.Nested.Node")))
        """
    ) == "true 1 1"

    # A weakdep extension registers its rules whichever package is loaded first, and only
    # once both are.
    ext_state = """
        ext = Base.get_extension(FixtureHost, :FixtureHostWeakExt)
        print(ext !== nothing, " ", $(count_for("FixtureHost.Node")))
    """
    @test probe("import MessagePassingRulesBase, FixtureHost; " * ext_state) == "false 0"
    @test probe("import MessagePassingRulesBase, FixtureHost, FixtureWeak; " * ext_state) == "true 1"
    @test probe("import MessagePassingRulesBase, FixtureWeak, FixtureHost; " * ext_state) == "true 1"
end
