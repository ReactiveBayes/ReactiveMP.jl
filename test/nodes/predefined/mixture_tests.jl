@testitem "nodes:MixtureNode" begin
    using ReactiveMP, BayesBase, ExponentialFamily, Test
    import ReactiveMP:
        Mixture,
        MixtureNode,
        MixtureNodeFactorisation,
        MixtureNodeFunctionalDependencies,
        RequireMarginalFunctionalDependencies,
        NodeInterface,
        IndexedNodeInterface,
        interfaceindex,
        interfaceindices,
        collect_factorisation,
        collect_functional_dependencies,
        collect_latest_marginals,
        collect_latest_messages,
        functional_dependencies,
        alias_interface,
        is_predefined_node,
        PredefinedNodeFunctionalForm,
        interfaces,
        sdtype,
        factornode,
        FactorNodeActivationOptions,
        activate!

    # Common interfaces and factorizations used by both test groups
    interfaces_list = [(:out, datavar()), (:switch, randomvar()), (:inputs, randomvar()), (:inputs, randomvar())]
    factorizations = [[:out], [:switch], [:inputs1], [:inputs2]]

    @testset "Type-level definitions" begin
        @test ReactiveMP.as_node_symbol(Mixture{2}) === :Mixture
        @test interfaces(Mixture{2}) == Val((:out, :switch, :inputs))
        @test alias_interface(Mixture{2}, 1, :foo) === :foo
        @test is_predefined_node(Mixture{2}) isa PredefinedNodeFunctionalForm
        @test sdtype(Mixture{2}) === Stochastic()
        @test collect_factorisation(Mixture{2}, nothing) isa MixtureNodeFactorisation
    end

    @testset "Construction and interface structure" begin
        node = factornode(Mixture, interfaces_list, factorizations)

        @test node isa MixtureNode{2}
        @test sdtype(node) == Stochastic()
        @test functionalform(node) == Mixture{2}
        @test getinterfaces(node) isa Tuple
        @test length(getinterfaces(node)) == 4

        @test node.out isa NodeInterface
        @test node.switch isa NodeInterface
        @test all(i -> i isa IndexedNodeInterface, node.inputs)

        @test interfaceindex(node, :out) == 1
        @test interfaceindex(node, :switch) == 2
        @test interfaceindex(node, :inputs) == 3
        @test_throws ErrorException interfaceindex(node, :invalid)
    end

    @testset "Interface indices" begin
        node = factornode(Mixture, interfaces_list, factorizations)

        @test interfaceindices(node, :out) == (1,)
        @test interfaceindices(node, (:out, :switch)) == (1, 2)
    end

    @testset "Collect functional dependencies" begin
        node = MixtureNode(
            NodeInterface(interfaces_list[1]...),
            NodeInterface(interfaces_list[2]...),
            (IndexedNodeInterface(1, NodeInterface(interfaces_list[3]...)), IndexedNodeInterface(2, NodeInterface(interfaces_list[4]...)))
        )

        @test collect_functional_dependencies(node, nothing) isa MixtureNodeFunctionalDependencies
        @test collect_functional_dependencies(node, MixtureNodeFunctionalDependencies()) isa MixtureNodeFunctionalDependencies
        @test collect_functional_dependencies(node, RequireMarginalFunctionalDependencies()) isa RequireMarginalFunctionalDependencies
        @test_throws ErrorException collect_functional_dependencies(node, :wrongtype)
    end

    @testset "Functional dependencies" begin
        node = factornode(Mixture, interfaces_list, factorizations)
        deps = MixtureNodeFunctionalDependencies()

        msg, marg = functional_dependencies(deps, node, node.out, 1)
        @test msg == (node.switch, node.inputs)
        @test marg == ()

        msg, marg = functional_dependencies(deps, node, node.switch, 2)
        @test msg == (node.out, node.inputs)
        @test marg == ()

        msg, marg = functional_dependencies(deps, node, node.inputs[1], 3)
        @test msg == (node.out, node.switch)
        @test marg == ()

        @test_throws ErrorException functional_dependencies(deps, node, node.out, 99)
    end

    @testset "RequireMarginalFunctionalDependencies variant" begin
        node = factornode(Mixture, interfaces_list, factorizations)
        deps = RequireMarginalFunctionalDependencies()

        msg, marg = functional_dependencies(deps, node, 1)
        @test length(msg) == 1
        @test length(marg) == 1

        msg, marg = functional_dependencies(deps, node, 2)
        @test length(msg) == 2
        @test isempty(marg)

        msg, marg = functional_dependencies(deps, node, 3)
        @test length(msg) == 1
        @test length(marg) == 1
    end

    @testset "Collect latest marginals" begin
        node = factornode(Mixture, interfaces_list, factorizations)
        deps1 = MixtureNodeFunctionalDependencies()
        deps2 = RequireMarginalFunctionalDependencies()

        val1, obs1 = collect_latest_marginals(deps1, node, ())
        @test val1 === nothing
        @test obs1 !== nothing

        switchiface = NodeInterface(:switch, randomvar())
        val2, obs2 = collect_latest_marginals(deps2, node, (switchiface,))
        @test val2 isa Val
        @test obs2 !== nothing
    end
end
