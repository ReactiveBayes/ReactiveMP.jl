@testmodule ToyNodes begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: DefaultAlgorithm, AbstractAlgorithm

    struct MixtureVMP <: AbstractAlgorithm end
    import BayesBase

    struct Toy
        μ::Float64
        τ::Float64
    end
    BayesBase.logpdf(d::Toy, x) = -d.τ * (x - d.μ)^2

    @define_factor_node(
        node = Toy,
        type = Stochastic,
        interfaces = [:out, (:μ, aliases = [:mean, :m]), :τ],
    )

    struct Mixture end
    @define_factor_node(
        node = Mixture,
        type = Stochastic,
        interfaces = [:out, :switch, :inputs...],
        algorithm = MixtureVMP,
    )

    struct Params <: AbstractAlgorithm
        order::Int
    end
    struct TwoGroups end
    @define_factor_node(
        node = TwoGroups,
        type = Stochastic,
        interfaces = [:out, :a..., :b..., :x_y],
        algorithm = Params(3),
    )

    plus(a, b) = a + b
    @define_factor_node(node = plus, type = Deterministic, interfaces = [:out, :in1, :in2])

    # Groups that come in pairs, at least two members each, under mean-field only.
    struct Paired end
    @define_factor_node(
        node = Paired,
        type = Stochastic,
        interfaces = [:out, :switch, :m..., :p...],
        matched_groups = [(:m, :p)],
        min_group_length = 2,
        factorisation = :meanfield,
    )

    # A rule towards `in` reads the message on its own edge, seeded by default.
    struct Seeded end
    @define_factor_node(node = Seeded, type = Stochastic, interfaces = [:out, :in], initial_messages = [:in => 0.5])
end

@testitem "nodes:traits" tags = [:base] setup = [ToyNodes] begin
    using MessagePassingRulesBase: nodespec, interfaces, interface_groups, sdtype, default_algorithm, alias_interface,
        nodefunction, Stochastic, Deterministic, DefaultAlgorithm, NodeSpec, matched_groups, min_group_length, required_factorisation, initial_messages
    T = ToyNodes

    @test interfaces(T.Toy) === (:out, :μ, :τ)
    @test interface_groups(T.Toy) === ()
    @test sdtype(T.Toy) === Stochastic()
    @test default_algorithm(T.Toy) === DefaultAlgorithm()        # the default default

    @test interfaces(T.Mixture) === (:out, :switch, :inputs)
    @test interface_groups(T.Mixture) === (:inputs,)
    @test default_algorithm(T.Mixture) === T.MixtureVMP()

    # Several non-trailing interface_groups, an underscore in a name, an algorithm with parameters.
    @test interfaces(T.TwoGroups) === (:out, :a, :b, :x_y)
    @test interface_groups(T.TwoGroups) === (:a, :b)
    @test default_algorithm(T.TwoGroups) === T.Params(3)

    @test interfaces(T.plus) === (:out, :in1, :in2)
    @test sdtype(T.plus) === Deterministic()

    @test alias_interface(T.Toy, :mean) === :μ
    @test alias_interface(T.Toy, :m) === :μ
    @test alias_interface(T.Toy, :μ) === :μ
    @test_throws ArgumentError alias_interface(T.Toy, :nope)

    # What a node requires of the graph: nothing, unless declared.
    @test matched_groups(T.Mixture) === ()
    @test min_group_length(T.Mixture) == 1
    @test required_factorisation(T.Mixture) === :any
    @test matched_groups(T.Paired) === ((:m, :p),)
    @test min_group_length(T.Paired) == 2
    @test required_factorisation(T.Paired) === :meanfield
    @test contains(sprint(show, MIME("text/plain"), nodespec(T.Paired)), "matched groups:    m = p")
    @test initial_messages(T.Seeded) == (:in => 0.5,)
    @test initial_messages(T.Mixture) === ()
    @test contains(sprint(show, MIME("text/plain"), nodespec(T.Seeded)), "initial messages:  in")
    @test !contains(sprint(show, MIME("text/plain"), nodespec(T.Mixture)), "matched groups")

    spec = nodespec(T.Toy)
    @test spec isa NodeSpec
    @test spec.node === T.Toy
    @test endswith(String(spec.file), "node_tests.jl")
end

@testitem "nodes:nodefunction" tags = [:base] setup = [ToyNodes] begin
    using MessagePassingRulesBase: nodefunction
    T = ToyNodes
    f = nodefunction(T.Toy)
    @test f(out = 1.0, μ = 0.5, τ = 2.0) == -2.0 * (1.0 - 0.5)^2
    # Only stochastic nodes without interface_groups have one; a group has no positional meaning.
    @test_throws MethodError nodefunction(T.Mixture)
    @test_throws MethodError nodefunction(T.plus)
end

@testitem "nodes:registry" tags = [:base] setup = [ToyNodes] begin
    using MessagePassingRulesBase: registered_nodes
    T = ToyNodes
    ours = filter(spec -> parentmodule(spec.node isa Type ? spec.node : typeof(spec.node)) === T, registered_nodes())
    @test length(ours) == 6
end

@testitem "nodes:malformed" tags = [:base] begin
    using MessagePassingRulesBase

    function expansion_error(ex)
        err = try
            macroexpand(@__MODULE__, ex)
            nothing
        catch e
            e isa LoadError ? e.error : e
        end
        return err === nothing ? "" : sprint(showerror, err)
    end

    @test contains(expansion_error(:(@define_factor_node(type = Stochastic, interfaces = [:out]))), "`node` is required")
    @test contains(expansion_error(:(@define_factor_node(node = X, interfaces = [:out]))), "`type` is required")
    @test contains(expansion_error(:(@define_factor_node(node = X, type = Stochastic))), "`interfaces` is required")
    @test contains(expansion_error(:(@define_factor_node(node = X, type = Stochastic, interfaces = [:out], colour = 1))), "unknown keyword `colour`")
    @test contains(expansion_error(:(@define_factor_node(X, type = Stochastic, interfaces = [:out]))), "keyword")
    @test contains(expansion_error(:(@define_factor_node(node = X, type = Random, interfaces = [:out]))), "Stochastic or Deterministic")
    @test contains(expansion_error(:(@define_factor_node(node = X, type = Stochastic, interfaces = [:out, :μ, :μ]))), "duplicate interface `μ`")
    @test contains(expansion_error(:(@define_factor_node(node = X, type = Stochastic, interfaces = [:out, μ]))), "must be a symbol")
    @test contains(expansion_error(:(@define_factor_node(node = X, type = Stochastic, interfaces = [:out, (:μ, aliases = [:out])]))), "alias `out`")

    grouped(keyword) = expansion_error(Expr(:macrocall, Symbol("@define_factor_node"), nothing, :(node = X), :(type = Stochastic), :(interfaces = [:out, :m..., :p...]), keyword))
    @test contains(grouped(:(matched_groups = [(:m, :q)])), "names `q`, which is not a group")
    @test contains(grouped(:(matched_groups = [(:m,)])), "two or more distinct groups")
    @test contains(grouped(:(matched_groups = [(:m, :m)])), "two or more distinct groups")
    @test contains(grouped(:(matched_groups = (:m, :p))), "must be a vector of tuples")
    @test contains(grouped(:(min_group_length = -1)), "non-negative integer")
    # A group that may be empty, as DiscreteTransition's `T`.
    @test grouped(:(min_group_length = 0)) == ""
    @test contains(grouped(:(factorisation = :structured)), ":any or :meanfield")
    @test contains(expansion_error(:(@define_factor_node(node = X, type = Stochastic, interfaces = [:out, :x], min_group_length = 2))), "needs a group")
    @test grouped(:(matched_groups = [(:m, :p)])) == ""
    @test contains(grouped(:(initial_messages = [:q => 1.0])), "names `q`, which is not an interface")
    @test contains(grouped(:(initial_messages = [:m => 1.0])), "`m` is a group")
    @test contains(grouped(:(initial_messages = [:out => 1.0, :out => 2.0])), "more than once")
    @test contains(grouped(:(initial_messages = (:out => 1.0,))), "must be a vector of pairs")
    @test contains(expansion_error(:(@define_factor_node(node = X, type = Deterministic, interfaces = [:out, :x], factorisation = :meanfield))), "cannot require")
end

@testitem "nodes:getnodefn" tags = [:base] begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: AbstractAlgorithm, RuleArgs, RuleContext, Target, IndexedTarget, getnodefn

    # The base package only declares `getnodefn`; an engine's node implements it. Here a toy
    # engine node carries a forward function with its static inputs already folded in. A known
    # inverse is not the node's: it belongs to the algorithm, and a rule reads it from `algo`.
    struct DeltaToy end
    @define_factor_node(node = DeltaToy, type = Deterministic, interfaces = [:out, :in...])
    struct EngineNode{F}
        f::F
    end
    MessagePassingRulesBase.getnodefn(node::EngineNode, ::Target{:out}) = node.f

    struct Point{I} <: AbstractAlgorithm
        inverses::I
    end
    @define_message_update_rule(
        node = DeltaToy, target = :out, algorithm = Point, ctx = (:node,),
        args = (m[:in...]::Float64,),
        body = (ctx, args) -> getnodefn(ctx.node, Target(:out))(args.m[:in]...),
    )
    @define_message_update_rule(
        node = DeltaToy, target = (:in, k), algorithm = Point,
        args = (m[:out]::Float64,),
        body = (algo, args) -> algo.inverses[k](args.m[:out]),
    )

    algorithm = Point((z -> z - 2, z -> z / 2))
    ctx = RuleContext(node = EngineNode((x, y) -> x + 2y))
    @test getresult(message_passing_rule(DeltaToy, Target(:out), algorithm, RuleArgs(m = (in = (1.0, 3.0),)), ctx)) == 7.0
    @test getresult(message_passing_rule(DeltaToy, IndexedTarget(:in, 2), algorithm, RuleArgs(m = (out = 7.0,)))) == 3.5

    # Declared, with no methods of its own.
    @test isempty(methods(getnodefn, Tuple{Any, Any}, MessagePassingRulesBase))
    @test_throws MethodError getnodefn(nothing, Target(:out))
end
