module ReactiveMPNodeTest

using Test
using ReactiveMP
using Distributions

@testset "FactorNode" begin
    @testset "Common" begin
        @test ReactiveMP.as_node_functional_form(() -> nothing) === ReactiveMP.ValidNodeFunctionalForm()
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

        @node CustomStochasticNode Stochastic [out, (x, aliases = [xx]), (y, aliases = [yy]), z]

        @test ReactiveMP.interface_get_index(Val{:CustomStochasticNode}, Val{:out}) === 1
        @test ReactiveMP.interface_get_index(Val{:CustomStochasticNode}, Val{:x}) === 2
        @test ReactiveMP.interface_get_index(Val{:CustomStochasticNode}, Val{:y}) === 3
        @test ReactiveMP.interface_get_index(Val{:CustomStochasticNode}, Val{:z}) === 4

        @test ReactiveMP.interface_get_name(Val{:CustomStochasticNode}, Val{:out}) === :out
        @test ReactiveMP.interface_get_name(Val{:CustomStochasticNode}, Val{:x}) === :x
        @test ReactiveMP.interface_get_name(Val{:CustomStochasticNode}, Val{:y}) === :y
        @test ReactiveMP.interface_get_name(Val{:CustomStochasticNode}, Val{:z}) === :z

        @test ReactiveMP.interface_get_name(Val{:CustomStochasticNode}, Val{:xx}) === :x
        @test ReactiveMP.interface_get_name(Val{:CustomStochasticNode}, Val{:yy}) === :y

        @test ReactiveMP.interface_get_name(Val{:CustomStochasticNode}, Val{1}) === :out
        @test ReactiveMP.interface_get_name(Val{:CustomStochasticNode}, Val{2}) === :x
        @test ReactiveMP.interface_get_name(Val{:CustomStochasticNode}, Val{3}) === :y
        @test ReactiveMP.interface_get_name(Val{:CustomStochasticNode}, Val{4}) === :z

        @test ReactiveMP.collect_factorisation(CustomStochasticNode, ((1,), (2,), (3,), (4,))) ===
              ((1,), (2,), (3,), (4,))
        @test ReactiveMP.collect_factorisation(CustomStochasticNode, ((1, 2), (3,), (4,))) === ((1, 2), (3,), (4,))
        @test ReactiveMP.collect_factorisation(CustomStochasticNode, ((1, 2, 3), (4,))) === ((1, 2, 3), (4,))
        @test ReactiveMP.collect_factorisation(CustomStochasticNode, ((1, 2, 3), (4,))) === ((1, 2, 3), (4,))
        @test ReactiveMP.collect_factorisation(CustomStochasticNode, ((1, 2, 3, 4),)) === ((1, 2, 3, 4),)

        @test ReactiveMP.collect_factorisation(CustomStochasticNode, MeanField()) === ((1,), (2,), (3,), (4,))
        @test ReactiveMP.collect_factorisation(CustomStochasticNode, FullFactorisation()) === ((1, 2, 3, 4),)

        @test sdtype(CustomStochasticNode) === Stochastic()

        cx = constvar(:cx, 1.0)
        cy = constvar(:cy, 1.0)
        cz = constvar(:cz, 1.0)

        model = FactorGraphModel()

        snode, svar = make_node(
            model,
            FactorNodeCreationOptions(MeanField(), nothing, nothing),
            CustomStochasticNode,
            AutoVar(:cout),
            cx,
            cy,
            cz
        )

        @test snode ∈ getnodes(model)
        @test svar ∈ getrandom(model)

        @test snode !== nothing
        @test typeof(svar) <: RandomVariable
        @test factorisation(snode) === ((1,), (2,), (3,), (4,))

        # Testing Deterministic node specification

        struct CustomDeterministicNode end

        CustomDeterministicNode(x, y, z) = x + y + z

        @node CustomDeterministicNode Deterministic [out, (x, aliases = [xx]), (y, aliases = [yy]), z]

        @test ReactiveMP.interface_get_index(Val{:CustomDeterministicNode}, Val{:out}) === 1
        @test ReactiveMP.interface_get_index(Val{:CustomDeterministicNode}, Val{:x}) === 2
        @test ReactiveMP.interface_get_index(Val{:CustomDeterministicNode}, Val{:y}) === 3
        @test ReactiveMP.interface_get_index(Val{:CustomDeterministicNode}, Val{:z}) === 4

        @test ReactiveMP.interface_get_name(Val{:CustomDeterministicNode}, Val{:out}) === :out
        @test ReactiveMP.interface_get_name(Val{:CustomDeterministicNode}, Val{:x}) === :x
        @test ReactiveMP.interface_get_name(Val{:CustomDeterministicNode}, Val{:y}) === :y
        @test ReactiveMP.interface_get_name(Val{:CustomDeterministicNode}, Val{:z}) === :z

        @test ReactiveMP.interface_get_name(Val{:CustomDeterministicNode}, Val{:xx}) === :x
        @test ReactiveMP.interface_get_name(Val{:CustomDeterministicNode}, Val{:yy}) === :y

        @test ReactiveMP.interface_get_name(Val{:CustomDeterministicNode}, Val{1}) === :out
        @test ReactiveMP.interface_get_name(Val{:CustomDeterministicNode}, Val{2}) === :x
        @test ReactiveMP.interface_get_name(Val{:CustomDeterministicNode}, Val{3}) === :y
        @test ReactiveMP.interface_get_name(Val{:CustomDeterministicNode}, Val{4}) === :z

        @test ReactiveMP.collect_factorisation(CustomDeterministicNode, ((1,), (2,), (3,), (4,))) === ((1, 2, 3, 4),)
        @test ReactiveMP.collect_factorisation(CustomDeterministicNode, ((1, 2), (3,), (4,))) === ((1, 2, 3, 4),)
        @test ReactiveMP.collect_factorisation(CustomDeterministicNode, ((1, 2, 3), (4,))) === ((1, 2, 3, 4),)
        @test ReactiveMP.collect_factorisation(CustomDeterministicNode, ((1, 2, 3), (4,))) === ((1, 2, 3, 4),)
        @test ReactiveMP.collect_factorisation(CustomDeterministicNode, ((1, 2, 3, 4),)) === ((1, 2, 3, 4),)

        @test ReactiveMP.collect_factorisation(CustomDeterministicNode, MeanField()) === ((1, 2, 3, 4),)
        @test ReactiveMP.collect_factorisation(CustomDeterministicNode, FullFactorisation()) === ((1, 2, 3, 4),)

        @test sdtype(CustomDeterministicNode) === Deterministic()

        cx = constvar(:cx, 1.0)
        cy = constvar(:cy, 1.0)
        cz = constvar(:cz, 1.0)

        model = FactorGraphModel()

        snode, svar = make_node(
            model,
            FactorNodeCreationOptions(MeanField(), nothing, nothing),
            CustomDeterministicNode,
            AutoVar(:cout),
            cx,
            cy,
            cz
        )

        @test svar ∈ getconstant(model)

        @test snode === nothing
        @test typeof(svar) <: ConstVariable

        # Testing expected exceptions

        struct DummyStruct end

        @test_throws Exception eval(:(@node DummyStruct NotStochasticAndNotDeterministic [out, in, x]))
        @test_throws Exception eval(:(@node DummyStruct Stochastic [1, in, x]))
        @test_throws Exception eval(:(@node DummyStruct Stochastic [(1, aliases = [out]), in, x]))
        @test_throws Exception eval(:(@node DummyStruct Stochastic [(out, aliases = [out]), in, x]))
        @test_throws Exception eval(:(@node DummyStruct Stochastic [(out, aliases = [1]), in, x]))
        @test_throws Exception eval(:(@node DummyStruct Stochastic []))
    end

    @testset "sdtype of an arbitrary distribution is Stochastic" begin
        struct DummyDistribution <: Distribution{Univariate, Continuous} end

        @test sdtype(DummyDistribution) === Stochastic()
    end

    @testset "make_node throws on Unknown distribution type" begin
        struct DummyDistribution <: Distribution{Univariate, Continuous} end

        @test_throws ErrorException ReactiveMP.make_node(
            FactorGraphModel(),
            FactorNodeCreationOptions(),
            DummyDistribution,
            AutoVar(:θ)
        )
        @test_throws ErrorException ReactiveMP.make_node(
            FactorGraphModel(),
            FactorNodeCreationOptions(),
            DummyDistribution,
            randomvar(:θ)
        )
    end
end

end
