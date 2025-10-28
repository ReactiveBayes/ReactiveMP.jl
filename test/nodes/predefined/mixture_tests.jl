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
