module NotNodeTest

using Test
using ReactiveMP
using Random
import ReactiveMP: @test_rules

@testset "NotNode" begin
    @testset "Creation" begin
        node = make_node(NOT)

        @test functionalform(node) === NOT
        @test sdtype(node) === Stochastic()
        @test name.(interfaces(node)) === (:out, :in)
        @test factorisation(node) === ((1, 2),)
        @test metadata(node) === ProbitMeta(32)

        node = make_node(NOT, FactorNodeCreationOptions(nothing, 1, nothing))

        @test functionalform(node) === NOT
        @test sdtype(node) === Stochastic()
        @test name.(interfaces(node)) === (:out, :in)
        @test factorisation(node) === ((1, 2),)
        @test metadata(node) === 1
    end
end
end