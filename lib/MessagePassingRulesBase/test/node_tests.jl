@testmodule ToyNodes begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: BP, VMP, AbstractAlgorithm
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
        algorithm = VMP,
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
end

@testitem "nodes:traits" tags = [:base] setup = [ToyNodes] begin
    using MessagePassingRulesBase: nodespec, interfaces, groups, sdtype, default_algorithm, alias_interface,
        nodefunction, Stochastic, Deterministic, BP, VMP, NodeSpec
    T = ToyNodes

    @test interfaces(T.Toy) === (:out, :μ, :τ)
    @test groups(T.Toy) === ()
    @test sdtype(T.Toy) === Stochastic()
    @test default_algorithm(T.Toy) === BP()        # the default default

    @test interfaces(T.Mixture) === (:out, :switch, :inputs)
    @test groups(T.Mixture) === (:inputs,)
    @test default_algorithm(T.Mixture) === VMP()

    # Several non-trailing groups, an underscore in a name, an algorithm with parameters.
    @test interfaces(T.TwoGroups) === (:out, :a, :b, :x_y)
    @test groups(T.TwoGroups) === (:a, :b)
    @test default_algorithm(T.TwoGroups) === T.Params(3)

    @test interfaces(T.plus) === (:out, :in1, :in2)
    @test sdtype(T.plus) === Deterministic()

    @test alias_interface(T.Toy, :mean) === :μ
    @test alias_interface(T.Toy, :m) === :μ
    @test alias_interface(T.Toy, :μ) === :μ
    @test_throws ArgumentError alias_interface(T.Toy, :nope)

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
    # Only stochastic nodes without groups have one; a group has no positional meaning.
    @test_throws MethodError nodefunction(T.Mixture)
    @test_throws MethodError nodefunction(T.plus)
end

@testitem "nodes:registry" tags = [:base] setup = [ToyNodes] begin
    using MessagePassingRulesBase: registered_nodes
    T = ToyNodes
    ours = filter(spec -> parentmodule(spec.node isa Type ? spec.node : typeof(spec.node)) === T, registered_nodes())
    @test length(ours) == 4
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
end
