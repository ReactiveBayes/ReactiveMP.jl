@testitem "FactorNode" begin
    using ReactiveMP, Rocket, BayesBase, Distributions

    @testset "Common" begin
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

    @testset "@node macro" begin

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

        @test_throws LoadError eval(:(@node DummyStruct Stochastic [out, (interface, aliases = [alias_with_underscore])]))
    end

    @testset "sdtype of an arbitrary distribution is Stochastic" begin
        struct DummyDistribution <: Distribution{Univariate, Continuous} end

        @test sdtype(DummyDistribution) === Stochastic()
    end
end