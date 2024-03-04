@testitem "GenericFactorNode constructor" begin
    import ReactiveMP: functionalform, getinterfaces, getinterface

    struct ArbitraryNodeType end

    function foo end

    @testset "functionalform" begin
        @test @inferred(functionalform(factornode(ArbitraryNodeType, (;)))) === ArbitraryNodeType
        @test @inferred(functionalform(factornode(foo, (;)))) === typeof(foo)
    end

    @testset "getinterfaces" for fform in (ArbitraryNodeType, foo)
        a = randomvar()
        b = datavar()
        c = constvar(1)

        let node = factornode(fform, (a = a, b = b, c = c))
            @test name.(getinterfaces(node)) == (:a, :b, :c)
            @test name(getinterface(node, 1)) == :a
            @test name(getinterface(node, 2)) == :b
            @test name(getinterface(node, 3)) == :c
        end
    end
end

@testitem "sdtype" begin
    @test ReactiveMP.as_node_functional_form(() -> nothing) === ReactiveMP.UndefinedNodeFunctionalForm()
    @test ReactiveMP.as_node_functional_form(2) === ReactiveMP.UndefinedNodeFunctionalForm()

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

@testitem "@node macro" begin

    # Testing Stochastic node specification

    struct CustomStochasticNode end

    @node CustomStochasticNode Stochastic [out, x, y, z] aliases = [(out, xx, yy, z)]

    @test ReactiveMP.sdtype(CustomStochasticNode) === Stochastic()
    @test ReactiveMP.correct_interfaces(CustomStochasticNode, (out = 1, xx = 2, yy = 3, z = 4)) === (out = 1, x = 2, y = 3, z = 4)

    # Testing stochastic function node specification

    function customstochasticnode end

    @node typeof(customstochasticnode) Stochastic [out, x, y, z] aliases = [(out, xx, yy, z)]

    @test ReactiveMP.sdtype(customstochasticnode) === Stochastic()
    @test ReactiveMP.correct_interfaces(customstochasticnode, (out = 1, xx = 2, yy = 3, z = 4)) === (out = 1, x = 2, y = 3, z = 4)

    # Testing Deterministic node specification

    struct CustomDeterministicNode end

    CustomDeterministicNode(x, y, z) = x + y + z

    @node CustomDeterministicNode Deterministic [out, x, y, z] aliases = [(out, xx, yy, z)]

    @test ReactiveMP.sdtype(CustomDeterministicNode) === Deterministic()
    @test ReactiveMP.correct_interfaces(CustomDeterministicNode, (out = 1, xx = 2, yy = 3, z = 4)) === (out = 1, x = 2, y = 3, z = 4)

    # Testing deterministic function node specification

    function customdeterministicnode end

    customdeterministicnode(x, y, z) = x + y + z

    @node typeof(customdeterministicnode) Deterministic [out, x, y, z] aliases = [(out, xx, yy, z)]

    @test ReactiveMP.sdtype(customdeterministicnode) === Deterministic()
    @test ReactiveMP.correct_interfaces(customdeterministicnode, (out = 1, xx = 2, yy = 3, z = 4)) === (out = 1, x = 2, y = 3, z = 4)

    # Testing expected exceptions

    struct DummyStruct end

    @test_throws Exception eval(:(@node DummyStruct NotStochasticAndNotDeterministic [out, in, x]))
    @test_throws Exception eval(:(@node DummyStruct Stochastic [1, in, x]))
    @test_throws Exception eval(:(@node DummyStruct Stochastic [p, in, x] aliases = [([z], y, x)]))
    @test_throws Exception eval(:(@node DummyStruct Stochastic [(out, aliases = [out]), in, x]))
    @test_throws Exception eval(:(@node DummyStruct Stochastic [(out, aliases = [1]), in, x]))
    @test_throws Exception eval(:(@node DummyStruct Stochastic []))
end

@testitem "sdtype of an arbitrary distribution is Stochastic" begin
    using Distributions

    struct DummyDistribution <: Distribution{Univariate, Continuous} end

    @test sdtype(DummyDistribution) === Stochastic()
end