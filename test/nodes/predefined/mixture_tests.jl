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

@testitem "nodes:MixtureNode:Extended" begin
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
        collect_latest_marginals,
        collect_latest_messages,
        interfaces,
        alias_interface,
        is_predefined_node,
        PredefinedNodeFunctionalForm

    @testset "Type-level definitions" begin
        @test ReactiveMP.as_node_symbol(Mixture{2}) === :Mixture
        @test interfaces(Mixture{2}) == Val((:out, :switch, :inputs))
        @test alias_interface(Mixture{2}, 1, :foo) === :foo
        @test is_predefined_node(Mixture{2}) isa PredefinedNodeFunctionalForm
        @test sdtype(Mixture{2}) === Stochastic()
        @test collect_factorisation(Mixture{2}, nothing) isa MixtureNodeFactorisation
    end

    # Construct a simple MixtureNode for reuse
    vinterfaces = [(:out, datavar()), (:switch, randomvar()), (:inputs, randomvar()), (:inputs, randomvar())]
    factorizations = [[:out], [:switch], [:inputs1], [:inputs2]]
    node = factornode(Mixture, vinterfaces, factorizations)

    @testset "Interface indices" begin
        # Single symbol
        @test interfaceindices(node, :out) == (1,)
        # Multiple symbols
        res = interfaceindices(node, (:out, :switch))
        @test res == (1, 2)
    end

    # TODO: collect_latest_messages

    @testset "Collect latest marginals" begin
        deps1 = MixtureNodeFunctionalDependencies()
        deps2 = RequireMarginalFunctionalDependencies()

        # Variant 1: no marginals
        val1, obs1 = collect_latest_marginals(deps1, node, ())
        @test val1 === nothing
        @test obs1 !== nothing

        # Variant 2: with switch marginal
        switchiface = NodeInterface(:switch, randomvar())
        val2, obs2 = collect_latest_marginals(deps2, node, (switchiface,))
        @test val2 isa Val
        @test obs2 !== nothing
    end
end
