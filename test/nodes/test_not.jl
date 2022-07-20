module NotNodeTest

using Test
using ReactiveMP
using Random
import ReactiveMP: @test_rules

@testset "NotNode" begin
    @testset "Creation" begin
        node = make_node(NOT)

        @test functionalform(node) === NOT
        @test sdtype(node) === Deterministic()
        @test name.(interfaces(node)) === (:out, :in)
        @test factorisation(node) === ((1, 2),)
        @test metadata(node) === nothing
    end
end
end
