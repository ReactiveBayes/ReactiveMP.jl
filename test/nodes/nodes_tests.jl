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
    

    @test isdeterministic(Deterministic()) === true
    @test isdeterministic(Deterministic) === true
    @test isdeterministic(Stochastic()) === false
    @test isdeterministic(Stochastic) === false
    @test isstochastic(Deterministic()) === false
    @test isstochastic(Deterministic) === false
    @test isstochastic(Stochastic()) === true
    @test isstochastic(Stochastic) === true

    @test sdtype(() -> nothing) === Deterministic()
    @test_throws MethodError sdtype(0)
end

@testitem "is_predefined_node" begin 
    import ReactiveMP: is_predefined_node, PredefinedNodeFunctionalForm, UndefinedNodeFunctionalForm

    @test is_predefined_node(() -> nothing) === UndefinedNodeFunctionalForm()
    @test is_predefined_node(2) === UndefinedNodeFunctionalForm()

    struct ArbitraryFactorNodeForIsPredefinedTest end

    @node ArbitraryFactorNodeForIsPredefinedTest Stochastic [ out, in ]

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

    @node DummyNodeForDocumentationStochastic Stochastic [out, x, (y, aliases = [ yy ])]

    @node DummyNodeForDocumentationDeterministic Deterministic [out, (x, aliases = [ xx, xxx ]), y]

    documentation = string(Base.doc(Base.Docs.Binding(ReactiveMP, :is_predefined_node)))

    @test occursin(r"DummyNodeForDocumentationStochastic.*Stochastic.*out, x, y \(or yy\)", documentation)
    @test occursin(r"DummyNodeForDocumentationDeterministic.*Deterministic.*out, x \(or xx, xxx\), y", documentation)
end