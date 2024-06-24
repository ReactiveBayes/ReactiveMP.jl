@testitem "GenericFactorNode constructor" begin
    import ReactiveMP: functionalform, getinterfaces, getinterface, name

    struct ArbitraryNodeType end

    @node ArbitraryNodeType Stochastic [a, b, c]

    function foo end

    @node typeof(foo) Deterministic [a, b, c]

    a = randomvar()
    b = datavar()
    c = constvar(1)

    @testset "functionalform" begin
        @test @inferred(functionalform(factornode(ArbitraryNodeType, [(:a, a), (:b, b), (:c, c)], ((1, 2, 3),)))) === ArbitraryNodeType
        @test @inferred(functionalform(factornode(foo, [(:a, a), (:b, b), (:c, c)], ((1, 2, 3),)))) === foo
    end

    @testset "getinterfaces" for fform in (ArbitraryNodeType, foo)
        let node = factornode(fform, [(:a, a), (:b, b), (:c, c)], ((1, 2, 3),))
            @test name.(getinterfaces(node)) == [:a, :b, :c]
            @test name(getinterface(node, 1)) == :a
            @test name(getinterface(node, 2)) == :b
            @test name(getinterface(node, 3)) == :c
        end
    end
end

@testitem "sdtype" begin
    using Distributions

    @test isdeterministic(Deterministic()) === true
    @test isdeterministic(Deterministic) === true
    @test isdeterministic(Stochastic()) === false
    @test isdeterministic(Stochastic) === false
    @test isstochastic(Deterministic()) === false
    @test isstochastic(Deterministic) === false
    @test isstochastic(Stochastic()) === true
    @test isstochastic(Stochastic) === true

    @test sdtype(() -> nothing) === Deterministic()
    @test sdtype(Normal(0.0, 1.0)) === Stochastic()

    @test_throws "Unknown if an object of type `Vector{Float64}` is stochastic or deterministic." sdtype([1.0, 2.0, 3.0])
    @test_throws "Unknown if an object of type `Matrix{Float64}` is stochastic or deterministic." sdtype([1.0 0.0; 0.0 1.0])
    @test_throws "Unknown if an object of type `Int64` is stochastic or deterministic." sdtype(0)
end

@testitem "is_predefined_node" begin
    import ReactiveMP: is_predefined_node, PredefinedNodeFunctionalForm, UndefinedNodeFunctionalForm

    @test is_predefined_node(() -> nothing) === UndefinedNodeFunctionalForm()
    @test is_predefined_node(2) === UndefinedNodeFunctionalForm()

    struct ArbitraryFactorNodeForIsPredefinedTest end

    @node ArbitraryFactorNodeForIsPredefinedTest Stochastic [out, in]

    @test is_predefined_node(ArbitraryFactorNodeForIsPredefinedTest) === PredefinedNodeFunctionalForm()
end

@testitem "@node macro" begin
    import ReactiveMP: alias_interface

    struct CustomStochasticNode end

    @node CustomStochasticNode Stochastic [out, (x, aliases = [xx]), (y, aliases = [yy]), z]

    function customstochasticnode end

    @node typeof(customstochasticnode) Stochastic [out, (x, aliases = [xx]), (y, aliases = [yy]), z]

    struct CustomDeterministicNode end

    CustomDeterministicNode(x, y, z) = x + y + z

    @node CustomDeterministicNode Deterministic [out, (x, aliases = [xx]), (y, aliases = [yy]), z]

    function customdeterministicnode end

    customdeterministicnode(x, y, z) = x + y + z

    @node typeof(customdeterministicnode) Deterministic [out, (x, aliases = [xx]), (y, aliases = [yy]), z]

    @test ReactiveMP.sdtype(CustomStochasticNode) === Stochastic()
    @test ReactiveMP.sdtype(customstochasticnode) === Stochastic()
    @test ReactiveMP.sdtype(CustomDeterministicNode) === Deterministic()
    @test ReactiveMP.sdtype(customdeterministicnode) === Deterministic()

    for node in [CustomStochasticNode, customstochasticnode, CustomDeterministicNode, customdeterministicnode]
        @test alias_interface(node, 1, :out) === :out
        @test alias_interface(node, 2, :x) === :x
        @test alias_interface(node, 2, :xx) === :x
        @test alias_interface(node, 3, :y) === :y
        @test alias_interface(node, 3, :yy) === :y
        @test_throws ErrorException alias_interface(node, 4, :out) === :x
        @test_throws ErrorException alias_interface(node, 4, :x) === :x
        @test_throws ErrorException alias_interface(node, 4, :y) === :x
        @test_throws ErrorException alias_interface(node, 4, :zz)
    end

    struct DummyStruct end

    @test_throws Exception eval(:(@node DummyStruct NotStochasticAndNotDeterministic [out, in, x]))
    @test_throws Exception eval(:(@node DummyStruct Stochastic [1, in, x]))
    @test_throws Exception eval(:(@node DummyStruct Stochastic [p, in, x] aliases = [([z], y, x)]))
    @test_throws Exception eval(:(@node DummyStruct Stochastic [(out, aliases = [1]), in, x]))
    @test_throws Exception eval(:(@node DummyStruct Stochastic []))
end

@testitem "sdtype of an arbitrary distribution is Stochastic" begin
    using Distributions

    struct DummyDistribution <: Distribution{Univariate, Continuous} end

    @test sdtype(DummyDistribution) === Stochastic()
end

# This is a limitation of the current implementation, which can be removed in the future
@testitem "@node macro (in the current implementation) should not support interface names with underscores" begin
    @test_throws "Node interfaces names (and aliases) must not contain `_` symbol in them, found in `c_d`" eval(
        quote
            struct DummyNode end

            @node DummyNode Stochastic [out, c_d]
        end
    )
    @test_throws "Node interfaces names (and aliases) must not contain `_` symbol in them, found in `d_b_a`" eval(
        quote
            struct DummyNode end

            @node DummyNode Stochastic [out, c, d_b_a]
        end
    )
    @test_throws "Node interfaces names (and aliases) must not contain `_` symbol in them, found in `c_d`" eval(
        quote
            struct DummyNode end

            @node DummyNode Stochastic [out, (c, aliases = [c_d])]
        end
    )
end

@testitem "@node macro should generate a documentation entry for a newly specified node" begin
    struct DummyNodeForDocumentationStochastic end
    struct DummyNodeForDocumentationDeterministic end

    @node DummyNodeForDocumentationStochastic Stochastic [out, x, (y, aliases = [yy])]

    @node DummyNodeForDocumentationDeterministic Deterministic [out, (x, aliases = [xx, xxx]), y]

    documentation = string(Base.doc(Base.Docs.Binding(ReactiveMP, :is_predefined_node)))

    @test occursin(r"DummyNodeForDocumentationStochastic.*Stochastic.*out, x, y \(or yy\)", documentation)
    @test occursin(r"DummyNodeForDocumentationDeterministic.*Deterministic.*out, x \(or xx, xxx\), y", documentation)
end

@testitem "Predefined nodes should check the arguments supplied" begin
    struct StochasticNodeWithThreeArguments end
    struct DeterministicNodeWithFourArguments end

    @node StochasticNodeWithThreeArguments Stochastic [out, x, y, z]
    @node DeterministicNodeWithFourArguments Deterministic [out, x, y, z, w]

    out = randomvar()
    x = randomvar()
    y = randomvar()
    z = randomvar()
    w = randomvar()

    @test factornode(StochasticNodeWithThreeArguments, [(:out, out), (:x, x), (:y, y), (:z, z)], ((1, 2, 3),)) isa ReactiveMP.FactorNode
    @test factornode(DeterministicNodeWithFourArguments, [(:out, out), (:x, x), (:y, y), (:z, z), (:w, w)], ((1, 2, 3, 4),)) isa ReactiveMP.FactorNode

    @test_throws r"At least one argument is required for a factor node. Got none for `.*StochasticNodeWithThreeArguments`" factornode(StochasticNodeWithThreeArguments, [], ())
    @test_throws r"At least one argument is required for a factor node. Got none for `.*DeterministicNodeWithFourArguments`" factornode(DeterministicNodeWithFourArguments, [], ())
    @test_throws r"Expected 3 input arguments for `.*StochasticNodeWithThreeArguments`, got 1: x" factornode(StochasticNodeWithThreeArguments, [(:out, out), (:x, x)], ((1,),))
    @test_throws r"Expected 3 input arguments for `.*StochasticNodeWithThreeArguments`, got 2: x, y" factornode(
        StochasticNodeWithThreeArguments, [(:out, out), (:x, x), (:y, y)], ((1, 2),)
    )
    @test_throws r"Expected 3 input arguments for `.*StochasticNodeWithThreeArguments`, got 4: x, y, z, w" factornode(
        StochasticNodeWithThreeArguments, [(:out, out), (:x, x), (:y, y), (:z, z), (:w, w)], ((1, 2, 3, 4),)
    )
    @test_throws r"Expected 4 input arguments for `.*DeterministicNodeWithFourArguments`, got 1: x" factornode(DeterministicNodeWithFourArguments, [(:out, out), (:x, x)], ((1,),))
    @test_throws r"Expected 4 input arguments for `.*DeterministicNodeWithFourArguments`, got 2: x, y" factornode(
        DeterministicNodeWithFourArguments, [(:out, out), (:x, x), (:y, y)], ((1, 2),)
    )
    @test_throws r"Expected 4 input arguments for `.*DeterministicNodeWithFourArguments`, got 3: x, y, z" factornode(
        DeterministicNodeWithFourArguments, [(:out, out), (:x, x), (:y, y), (:z, z)], ((1, 2, 3),)
    )

    @test_throws r"`.*StochasticNodeWithThreeArguments` has duplicate entry for interface `w`. Did you pass an array \(e.g. `x`\) instead of an array element \(e\.g\. `x\[i\]`\)\? Check your variable indices\." factornode(
        StochasticNodeWithThreeArguments, [(:out, out), (:x, x), (:w, w), (:w, w)], ((1, 2, 3, 4),)
    )
    @test_throws r"`.*StochasticNodeWithThreeArguments` has duplicate entry for interface `w`. Did you pass an array \(e.g. `x`\) instead of an array element \(e\.g\. `x\[i\]`\)\? Check your variable indices\." factornode(
        StochasticNodeWithThreeArguments, [(:out, out), (:x, x), (:y, y), (:z, z), (:w, w), (:w, w)], ((1, 2, 3, 4, 5, 6),)
    )
end

@testitem "Generic node construction checks should not allocate" begin
    import ReactiveMP: prepare_interfaces_check_adjacent_duplicates, prepare_interfaces_check_nonempty, prepare_interfaces_check_numarguments

    struct NodeForCheckDuplicatesTest end
    @node NodeForCheckDuplicatesTest Stochastic [out, x, y, z]

    out = randomvar()
    x = randomvar()
    y = randomvar()
    z = randomvar()

    interfaces = [(:out, out), (:x, x), (:y, y), (:z, z)]
    # compile first
    function foo(interfaces)
        prepare_interfaces_check_nonempty(NodeForCheckDuplicatesTest, interfaces)
        prepare_interfaces_check_adjacent_duplicates(NodeForCheckDuplicatesTest, interfaces)
        prepare_interfaces_check_numarguments(NodeForCheckDuplicatesTest, interfaces)
    end
    foo(interfaces)
    @test (@allocated(foo(interfaces)) == 0)
    @test (@allocations(foo(interfaces)) == 0)
end

@testitem "`@node` macro should generate the node function in all directions for `Stochastic` nodes" begin
    @testset "For a regular node a user needs to define a node function" begin
        struct DummyNodeForNodeFunction end

        @node DummyNodeForNodeFunction Stochastic [out, x, y, z]

        nodefunction = (out, x, y, z) -> out^4 + x^3 + y^2 + z

        ReactiveMP.nodefunction(::Type{DummyNodeForNodeFunction}) = (; out, x, y, z) -> nodefunction(out, x, y, z)

        for out in (-1, 1), x in (-1, 1), y in (-1, 1), z in (-1, 1)
            @test ReactiveMP.nodefunction(DummyNodeForNodeFunction, Val(:out), x = x, y = y, z = z)(out) ≈ nodefunction(out, x, y, z)
            @test ReactiveMP.nodefunction(DummyNodeForNodeFunction, Val(:x), out = out, y = y, z = z)(x) ≈ nodefunction(out, x, y, z)
            @test ReactiveMP.nodefunction(DummyNodeForNodeFunction, Val(:y), out = out, x = x, z = z)(y) ≈ nodefunction(out, x, y, z)
            @test ReactiveMP.nodefunction(DummyNodeForNodeFunction, Val(:z), out = out, x = x, y = y)(z) ≈ nodefunction(out, x, y, z)
        end
    end

    @testset "Distributions are the special case and they simple call the `logpdf` as their node function" begin
        using Distributions

        struct DummyNodeForNodeFunctionAsDistribution <: Distributions.ContinuousUnivariateDistribution
            mean
            var
        end

        @node DummyNodeForNodeFunctionAsDistribution Stochastic [out, mean, var]

        Distributions.logpdf(node::DummyNodeForNodeFunctionAsDistribution, out) = Distributions.logpdf(Distributions.Normal(node.mean, node.var), out)

        nodefunction = (out, mean, var) -> Distributions.logpdf(Distributions.Normal(mean, var), out)

        for out in (-1, 1), mean in (-1, 1), var in (1, 2)
            @test ReactiveMP.nodefunction(DummyNodeForNodeFunctionAsDistribution, Val(:out), mean = mean, var = var)(out) ≈ nodefunction(out, mean, var)
            @test ReactiveMP.nodefunction(DummyNodeForNodeFunctionAsDistribution, Val(:mean), out = out, var = var)(mean) ≈ nodefunction(out, mean, var)
            @test ReactiveMP.nodefunction(DummyNodeForNodeFunctionAsDistribution, Val(:var), out = out, mean = mean)(var) ≈ nodefunction(out, mean, var)
        end
    end
end