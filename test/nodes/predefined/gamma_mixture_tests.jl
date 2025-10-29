@testitem "GammaShapeLikelihood insupport" begin
    using BayesBase
    import ReactiveMP: GammaShapeLikelihood

    for p in [1.0, 2.0, 3.0], γ in [1.0, 2.0, 3.0], s in [1.0, 2.0]
        @test insupport(GammaShapeLikelihood(p, γ), s)
        @test !insupport(GammaShapeLikelihood(p, γ), -s)
    end
end

@testitem "nodes:GammaMixtureNode" begin
    using ReactiveMP, BayesBase, ExponentialFamily, Test

    import ReactiveMP:
        GammaMixtureNode,
        GammaMixtureNodeFactorisation,
        GammaMixtureNodeFunctionalDependencies,
        GammaShapeLikelihood,
        ManyOf,
        NodeInterface,
        IndexedNodeInterface,
        interfaceindex,
        collect_functional_dependencies,
        collect_latest_marginals
    import SpecialFunctions: loggamma
    interfaces = [(:out, datavar()), (:switch, randomvar()), (:a, randomvar()), (:a, randomvar()), (:b, randomvar()), (:b, randomvar())]
    factorizations = [[:out], [:switch], [:a1], [:a2], [:b1], [:b2]]

    @testset "Construction and interface structure" begin
        node = factornode(GammaMixture, interfaces, factorizations)

        @test node isa GammaMixtureNode{2}
        @test sdtype(node) == Stochastic()
        @test functionalform(node) == GammaMixture{2}
        @test getinterfaces(node) isa Tuple
        @test length(getinterfaces(node)) == 6

        @test node.out isa NodeInterface
        @test node.switch isa NodeInterface
        @test all(i -> i isa IndexedNodeInterface, node.as)
        @test all(i -> i isa IndexedNodeInterface, node.bs)

        # index mapping
        @test interfaceindex(node, :out) == 1
        @test interfaceindex(node, :switch) == 2
        @test interfaceindex(node, :a) == 3
        @test interfaceindex(node, :b) == 4
    end

    @testset "Construction errors" begin
        # not enough a/b
        @test_throws ErrorException factornode(GammaMixture, [(:out, :x), (:switch, :z), (:a, :a1), (:b, :b1)], [[:out], [:switch], [:a1], [:b1]])
        # mismatch count
        @test_throws ErrorException factornode(GammaMixture, [(:out, :x), (:switch, :z), (:a, :a1), (:a, :a2), (:b, :b1)], [[:out], [:switch], [:a1], [:a2], [:b1]])
        # wrong factorization
        @test_throws ErrorException factornode(GammaMixture, [(:out, :x), (:switch, :z), (:a, :a1), (:a, :a2), (:b, :b1), (:b, :b2)], [[:out, :switch]])
    end

    @testset "Functional dependencies" begin
        node = factornode(GammaMixture, interfaces, factorizations)
        deps = GammaMixtureNodeFunctionalDependencies()

        # out dependencies
        msg_deps, marg_deps = functional_dependencies(deps, node, node.out, 1)
        @test msg_deps == ()
        @test length(marg_deps) == 3  # (switch, as, bs)

        # switch dependencies
        msg_deps, marg_deps = functional_dependencies(deps, node, node.switch, 2)
        @test length(marg_deps) == 3  # (out, as, bs)

        # a dependencies
        msg_deps, marg_deps = functional_dependencies(deps, node, node.as[1], 3)
        @test length(marg_deps) == 3  # (out, switch, b_i)

        # b dependencies
        msg_deps, marg_deps = functional_dependencies(deps, node, node.bs[1], 5)
        @test length(marg_deps) == 3  # (out, switch, a_i)

        # invalid index
        @test_throws ErrorException functional_dependencies(deps, node, node.out, 99)
    end

    @testset "Collect functional dependencies" begin
        node = GammaMixtureNode(
            NodeInterface(interfaces[1]...),
            NodeInterface(interfaces[2]...),
            (IndexedNodeInterface(1, NodeInterface(interfaces[3]...)), IndexedNodeInterface(2, NodeInterface(interfaces[4]...))),
            (IndexedNodeInterface(1, NodeInterface(interfaces[5]...)), IndexedNodeInterface(2, NodeInterface(interfaces[6]...)))
        )

        @test collect_functional_dependencies(node, nothing) isa GammaMixtureNodeFunctionalDependencies
        @test collect_functional_dependencies(node, GammaMixtureNodeFunctionalDependencies()) isa GammaMixtureNodeFunctionalDependencies
        @test_throws ErrorException collect_functional_dependencies(node, :wrongtype)
    end

    @testset "Collect latest marginals (arity check)" begin
        deps = GammaMixtureNodeFunctionalDependencies()
        node = GammaMixtureNode(
            NodeInterface(interfaces[1]...),
            NodeInterface(interfaces[2]...),
            (IndexedNodeInterface(1, NodeInterface(interfaces[3]...)), IndexedNodeInterface(2, NodeInterface(interfaces[4]...))),
            (IndexedNodeInterface(1, NodeInterface(interfaces[5]...)), IndexedNodeInterface(2, NodeInterface(interfaces[6]...)))
        )

        # First overload: (out, as, bs)
        marg_names, marg_obs = collect_latest_marginals(deps, node, (node.out, node.as, node.bs))
        @test marg_names isa Val
        @test !isnothing(marg_obs)

        # Second overload: (out, switch, var)
        marg_names, marg_obs = collect_latest_marginals(deps, node, (node.out, node.switch, node.as[1]))
        @test marg_names isa Val
        @test !isnothing(marg_obs)
    end

    @testset "GammaShapeLikelihood basic checks" begin
        d1 = GammaShapeLikelihood(1.0, 2.0)
        d2 = GammaShapeLikelihood(2.0, 3.0)
        @test params(d1) == (1.0, 2.0)
        @test logpdf(d1, 1.0) ≈ 2.0 * 1.0 - 1.0 * loggamma(1.0)
        prod_d = prod(PreserveTypeProd(Distribution), d1, d2)
        @test prod_d isa GammaShapeLikelihood
        @test prod_d.p == 3.0
        @test prod_d.γ == 5.0
    end
end

@testitem "AverageEnergy" begin
    using ReactiveMP, BayesBase, ExponentialFamily, Random, Test

    import ReactiveMP: ManyOf, GammaMixture
    import ExponentialFamily: NormalMeanVariance, NormalMeanPrecision, GammaShapeRate

    @testset "GammaMixture AverageEnergy" begin
        q_out = GammaShapeRate(1.0, 1.0)
        q_switch = Categorical([0.2, 0.8])
        q_a = (GammaShapeRate(2.0, 3.0), GammaShapeRate(4.0, 5.0))
        q_b = (GammaShapeRate(1.5, 2.5), GammaShapeRate(3.5, 4.5))

        marginals = (
            Marginal(q_out, false, false, nothing),
            Marginal(q_switch, false, false, nothing),
            ManyOf(map(q -> Marginal(q, false, false, nothing), q_a)),
            ManyOf(map(q -> Marginal(q, false, false, nothing), q_b))
        )

        # @average_energy GammaMixture (q_out::Any, q_switch::Any, q_a::ManyOf{N, Any}, q_b::ManyOf{N, GammaShapeRate})
        z = probvec(q_switch)
        ref_val =
            z[1] * score(AverageEnergy(), GammaShapeRate, Val{(:out, :α, :β)}(), map((q) -> Marginal(q, false, false, nothing), (q_out, q_a[1], q_b[1])), nothing) +
            z[2] * score(AverageEnergy(), GammaShapeRate, Val{(:out, :α, :β)}(), map((q) -> Marginal(q, false, false, nothing), (q_out, q_a[2], q_b[2])), nothing)

        @test score(AverageEnergy(), GammaMixture, Val{(:out, :switch, :a, :b)}(), marginals, nothing) ≈ ref_val
    end
end

@testitem "GammaMixture: type-level utilities" begin
    using ReactiveMP, Test
    import ReactiveMP: GammaMixture, GammaMixtureNodeFactorisation, as_node_symbol, interfaces, alias_interface, sdtype, collect_factorisation

    @test as_node_symbol(GammaMixture{2}) === :GammaMixture
    @test interfaces(GammaMixture{2}) === Val((:out, :switch, :a, :b))
    @test alias_interface(GammaMixture, 1, :a) === :a
    @test sdtype(GammaMixture{2}) == Stochastic()

    # collect_factorisation should always return GammaMixtureNodeFactorisation
    fact = collect_factorisation(GammaMixture{2}, :anything)
    @test fact isa GammaMixtureNodeFactorisation
end

@testitem "GammaMixtureNode: interfaceindices and unknown interface" begin
    using ReactiveMP, Test
    import ReactiveMP: GammaMixtureNode, NodeInterface, IndexedNodeInterface, interfaceindex, interfaceindices, functionalform

    # minimal fake interfaces
    interfaces = [(:out, datavar()), (:switch, randomvar()), (:a, randomvar()), (:a, randomvar()), (:b, randomvar()), (:b, randomvar())]
    node = GammaMixtureNode(
        NodeInterface(interfaces[1]...),
        NodeInterface(interfaces[2]...),
        (IndexedNodeInterface(1, NodeInterface(interfaces[3]...)), IndexedNodeInterface(2, NodeInterface(interfaces[4]...))),
        (IndexedNodeInterface(1, NodeInterface(interfaces[5]...)), IndexedNodeInterface(2, NodeInterface(interfaces[6]...)))
    )

    # Single symbol version
    @test interfaceindices(node, :out) == (1,)
    @test interfaceindices(node, :switch) == (2,)

    # Multiple symbols version
    syms = (:out, :switch, :a, :b)
    idxs = interfaceindices(node, syms)
    @test idxs == map(s -> interfaceindex(node, s), syms)

    # Unknown interface must throw and contain node functionalform in message
    @test_throws ErrorException interfaceindex(node, :nonexistent)
end

@testitem "GammaMixture: mismatch count error message" begin
    using ReactiveMP, Test
    import ReactiveMP: factornode, GammaMixture

    # mismatched counts of a and b
    @test_throws ErrorException factornode(GammaMixture, [(:out, :x), (:switch, :z), (:a, :a1), (:b, :b1), (:b, :b2)], [[:out], [:switch], [:a1], [:b1], [:b2]])
    @test_throws ErrorException factornode(
        GammaMixture, [(:out, :x), (:switch, :z), (:a, :a1), (:a, :a2), (:a, :a3), (:b, :b1), (:b, :b2)], [[:out], [:switch], [:a1, :a2, :a3], [:b1], [:b2]]
    )
end

@testitem "GammaMixtureNodeFunctionalDependencies: collect_latest_messages empty tuple" begin
    using ReactiveMP, Test
    import ReactiveMP: GammaMixtureNodeFunctionalDependencies, GammaMixtureNode, NodeInterface, IndexedNodeInterface, collect_latest_messages

    interfaces = [(:out, datavar()), (:switch, randomvar()), (:a, randomvar()), (:a, randomvar()), (:b, randomvar()), (:b, randomvar())]
    node = GammaMixtureNode(
        NodeInterface(interfaces[1]...),
        NodeInterface(interfaces[2]...),
        (IndexedNodeInterface(1, NodeInterface(interfaces[3]...)), IndexedNodeInterface(2, NodeInterface(interfaces[4]...))),
        (IndexedNodeInterface(1, NodeInterface(interfaces[5]...)), IndexedNodeInterface(2, NodeInterface(interfaces[6]...)))
    )

    deps = GammaMixtureNodeFunctionalDependencies()
    val, obs = collect_latest_messages(deps, node, ())
    @test val === nothing
    @test obs !== nothing
end

@testitem "GammaShapeLikelihood: support and prod rule" begin
    using ReactiveMP, BayesBase, Distributions, Test
    import ReactiveMP: GammaShapeLikelihood

    d = GammaShapeLikelihood(1.0, 2.0)

    # support should be (0.0, Inf)
    s = support(d)
    @test s isa Distributions.RealInterval
    @test s.lb == 0.0
    @test s.ub == Inf

    # default_prod_rule dispatch
    rule = BayesBase.default_prod_rule(GammaShapeLikelihood, GammaShapeLikelihood)
    @test rule == PreserveTypeProd(Distribution)
end
