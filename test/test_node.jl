module ReactiveMPNodeTest

using Test
using ReactiveMP
using Rocket
using Distributions

# Node specifications should be at top-level
struct CustomStochasticNode end

@node CustomStochasticNode Stochastic [out, (x, aliases = [xx]), (y, aliases = [yy]), z]

struct CustomDeterministicNode end

CustomDeterministicNode(x, y, z) = x + y + z

@node CustomDeterministicNode Deterministic [out, (x, aliases = [xx]), (y, aliases = [yy]), z]

struct DummyNodeCheckUniqueness end

@node DummyNodeCheckUniqueness Stochastic [a, b, c]

struct DummyNodeCheckFactorisationWarning end

@node DummyNodeCheckFactorisationWarning Stochastic [a, b, c]

@testset "FactorNode" begin
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

        @test ReactiveMP.collect_factorisation(CustomStochasticNode, ((1,), (2,), (3,), (4,))) === ((1,), (2,), (3,), (4,))
        @test ReactiveMP.collect_factorisation(CustomStochasticNode, ((1, 2), (3,), (4,))) === ((1, 2), (3,), (4,))
        @test ReactiveMP.collect_factorisation(CustomStochasticNode, ((1, 2, 3), (4,))) === ((1, 2, 3), (4,))
        @test ReactiveMP.collect_factorisation(CustomStochasticNode, ((1, 2, 3), (4,))) === ((1, 2, 3), (4,))
        @test ReactiveMP.collect_factorisation(CustomStochasticNode, ((1, 2, 3, 4),)) === ((1, 2, 3, 4),)

        @test ReactiveMP.collect_factorisation(CustomStochasticNode, MeanField()) === ((1,), (2,), (3,), (4,))
        @test ReactiveMP.collect_factorisation(CustomStochasticNode, FullFactorisation()) === ((1, 2, 3, 4),)

        @test sdtype(CustomStochasticNode) === Stochastic()

        # Testing Deterministic node specification

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

        # Check that same variables are not allowed
        sx = randomvar(:rx)
        sd = datavar(:rd, Float64)
        sc = constvar(:sc, 1.0)

        vs = (sx, sd, sc)

        for a in vs, b in vs, c in vs
            input = (a, b, c)
            if length(input) != length(Set(input))
                @test_throws ErrorException make_node(DummyNodeCheckUniqueness, FactorNodeCreationOptions(), a, b, c)
            end
        end

        # `make_node` must show a warning in case if factorisation include the `PointMass` distributed variables jointly with other variables
        for a in (datavar(:a, Float64), constvar(:a, 1.0)), b in (randomvar(:b),), c in (randomvar(:c),)
            @test_logs (:warn, r".*replace `q\(a, b, c\)` with `q\(a\)q\(\.\.\.\)`.*") make_node(
                DummyNodeCheckFactorisationWarning, FactorNodeCreationOptions(FullFactorisation(), nothing, nothing), a, b, c
            )
            @test_logs (:warn, r".*replace `q\(a, b, c\)` with `q\(a\)q\(\.\.\.\)`.*") make_node(
                DummyNodeCheckFactorisationWarning, FactorNodeCreationOptions(((1, 2, 3),), nothing, nothing), a, b, c
            )
            @test_logs (:warn, r".*replace `q\(a, b\)` with `q\(a\)q\(\.\.\.\)`.*") make_node(
                DummyNodeCheckFactorisationWarning, FactorNodeCreationOptions(((1, 2), (3,)), nothing, nothing), a, b, c
            )
        end

        # Testing expected exceptions

        struct DummyStruct end

        @test_throws Exception eval(:(@node DummyStruct NotStochasticAndNotDeterministic [out, in, x]))
        @test_throws Exception eval(:(@node DummyStruct Stochastic [1, in, x]))
        @test_throws Exception eval(:(@node DummyStruct Stochastic [(1, aliases = [out]), in, x]))
        @test_throws Exception eval(:(@node DummyStruct Stochastic [(out, aliases = [out]), in, x]))
        @test_throws Exception eval(:(@node DummyStruct Stochastic [(out, aliases = [1]), in, x]))
        @test_throws Exception eval(:(@node DummyStruct Stochastic []))

        @test_throws LoadError eval(:(@node DummyStruct Stochastic [out, interfaces_with_underscore]))
        @test_throws LoadError eval(:(@node DummyStruct Stochastic [out, (interface, aliases = [alias_with_underscore])]))
    end

    @testset "sdtype of an arbitrary distribution is Stochastic" begin
        struct DummyDistribution <: Distribution{Univariate, Continuous} end

        @test sdtype(DummyDistribution) === Stochastic()
    end
end

end
