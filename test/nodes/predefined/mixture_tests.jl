@testitem "nodes:MixtureNode" begin
    using ReactiveMP, BayesBase, ExponentialFamily, Test
    import ReactiveMP:
        MixtureNode,
        MixtureNodeFactorisation,
        MixtureNodeFunctionalDependencies,
        RequireMarginalFunctionalDependencies,
        NodeInterface,
        IndexedNodeInterface,
        interfaceindex,
        collect_functional_dependencies,
        collect_latest_marginals,
        collect_latest_messages,
        factornode,
        functional_dependencies,
        FactorNodeActivationOptions,
        activate!,
        Mixture,
        MixtureNode

    interfaces = [(:out, datavar()), (:switch, randomvar()), (:inputs, randomvar()), (:inputs, randomvar())]
    factorizations = [[:out], [:switch], [:inputs1], [:inputs2]]

    @testset "Construction and interface structure" begin
        node = factornode(Mixture, interfaces, factorizations)

        @test node isa MixtureNode{2}
        @test sdtype(node) == Stochastic()
        @test functionalform(node) == Mixture{2}
        @test getinterfaces(node) isa Tuple
        @test length(getinterfaces(node)) == 4  # out, switch, inputs...

        @test node.out isa NodeInterface
        @test node.switch isa NodeInterface
        @test all(i -> i isa IndexedNodeInterface, node.inputs)

        # index mapping
        @test interfaceindex(node, :out) == 1
        @test interfaceindex(node, :switch) == 2
        @test interfaceindex(node, :inputs) == 3
        @test_throws ErrorException interfaceindex(node, :invalid)
    end

    @testset "Collect functional dependencies" begin
        node = MixtureNode(
            NodeInterface(interfaces[1]...),
            NodeInterface(interfaces[2]...),
            (IndexedNodeInterface(1, NodeInterface(interfaces[3]...)), IndexedNodeInterface(2, NodeInterface(interfaces[4]...)))
        )

        @test collect_functional_dependencies(node, nothing) isa MixtureNodeFunctionalDependencies
        @test collect_functional_dependencies(node, MixtureNodeFunctionalDependencies()) isa MixtureNodeFunctionalDependencies
        @test collect_functional_dependencies(node, RequireMarginalFunctionalDependencies()) isa RequireMarginalFunctionalDependencies
        @test_throws ErrorException collect_functional_dependencies(node, :wrongtype)
    end

    @testset "Functional dependencies" begin
        node = factornode(Mixture, interfaces, factorizations)
        deps = MixtureNodeFunctionalDependencies()

        # out
        msg, marg = functional_dependencies(deps, node, node.out, 1)
        @test msg == (node.switch, node.inputs)
        @test marg == ()

        # switch
        msg, marg = functional_dependencies(deps, node, node.switch, 2)
        @test msg == (node.out, node.inputs)
        @test marg == ()

        # inputs
        msg, marg = functional_dependencies(deps, node, node.inputs[1], 3)
        @test msg == (node.out, node.switch)
        @test marg == ()

        # invalid
        @test_throws ErrorException functional_dependencies(deps, node, node.out, 99)
    end

    @testset "RequireMarginalFunctionalDependencies variant" begin
        node = factornode(Mixture, interfaces, factorizations)
        deps = RequireMarginalFunctionalDependencies()

        # out depends on inputs + marginal on switch
        msg, marg = functional_dependencies(deps, node, 1)
        @test length(msg) == 1
        @test length(marg) == 1

        # switch depends on out, inputs + no marginal
        msg, marg = functional_dependencies(deps, node, 2)
        @test length(msg) == 2
        @test isempty(marg)

        # input depends on out + marginal on switch
        msg, marg = functional_dependencies(deps, node, 3)
        @test length(msg) == 1
        @test length(marg) == 1
    end
end

@testitem "AverageEnergy" skip=true begin
    using ReactiveMP, BayesBase, ExponentialFamily, Random, Test

    import ReactiveMP: ManyOf, Mixture
    import ExponentialFamily: NormalMeanVariance, NormalMeanPrecision, GammaShapeRate
    import BayesBase: score

    @testset "Mixture AverageEnergy" begin
        q_out = NormalMeanVariance(0.0, 1.0)
        q_switch = Categorical([0.3, 0.7])
        q_m = (NormalMeanVariance(1.0, 2.0), NormalMeanVariance(3.0, 4.0))
        q_p = (GammaShapeRate(2.0, 3.0), GammaShapeRate(4.0, 5.0))

        marginals = (
            Marginal(q_out, false, false, nothing),
            Marginal(q_switch, false, false, nothing),
            ManyOf(map(q_m_ -> Marginal(q_m_, false, false, nothing), q_m)),
            ManyOf(map(q_p_ -> Marginal(q_p_, false, false, nothing), q_p))
        )

        z = probvec(q_switch)
        ref_val =
            z[1] * score(AverageEnergy(), NormalMeanPrecision, Val{(:out, :μ, :τ)}(), map((q) -> Marginal(q, false, false, nothing), (q_out, q_m[1], q_p[1])), nothing) +
            z[2] * score(AverageEnergy(), NormalMeanPrecision, Val{(:out, :μ, :τ)}(), map((q) -> Marginal(q, false, false, nothing), (q_out, q_m[2], q_p[2])), nothing)

        @test score(AverageEnergy(), Mixture, Val{(:out, :switch, :inputs)}(), marginals, nothing) ≈ ref_val
    end
end
