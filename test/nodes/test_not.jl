module NotNodeTest

using ReactiveMP, Random, BayesBase, ExponentialFamily

@testitem "NotNode" begin
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
